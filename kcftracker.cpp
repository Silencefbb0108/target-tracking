#include "stdafx.h"

#include "kcftracker.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"
#include "fhog.hpp"
#include "opencv2/imgproc/imgproc.hpp"  

using namespace cv;

// Constructor
KCFTracker::KCFTracker(bool hog)
{
    lambda = 0.0001;
    padding = 2.5;
    output_sigma_factor = 0.125;

    if (hog)
    {   
        interp_factor = 0.012; 
        sigma = 0.6;
        cell_size = 4;
        _hogfeatures = true;
    }
    else
    {   // RAW
        interp_factor = 0.075;
        sigma = 0.2;
        cell_size = 1;
        _hogfeatures = false;
    }
	template_size = 96;
    scale_step = 1.1; //Mark bigger or smaller
	scale_weight = 0.95; //Mark test scale_weight	
}

// Initialize tracker 
void KCFTracker::init(const cv::Rect &roi, cv::Mat image)
{
    _roi = roi;
    assert(roi.width >= 0 && roi.height >= 0);
    _tmpl = getFeatures(image, 1);//只有第一次初始化的时候，第二个形参才为1，对第一帧特征进行汉宁窗平滑
    _prob = createGaussianPeak(size_patch[0], size_patch[1]);//创建高斯峰，只有第一帧才用到
    _alphaf = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));////alphaf初始化
    train(_tmpl, 1.0); // train with initial frame
}

void KCFTracker::determine(cv::Mat image, float cx, float cy, cv::Point2f res)
{
    _roi.x = cx - _roi.width / 2.0f + ((float)res.x * cell_size * _scale_x);
    _roi.y = cy - _roi.height / 2.0f + ((float)res.y * cell_size * _scale_y);

    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;

    assert(_roi.width >= 0 && _roi.height >= 0);

    cv::Mat x = getFeatures(image, 0);//提取新的roi特征
    train(x, interp_factor);//训练得到新的滤波器
    return;
}

// Update position based on the new frame
cv::Rect KCFTracker::update(cv::Mat image)
{
    if (_roi.x + _roi.width <= 0)
        _roi.x = -_roi.width + 1;
    if (_roi.y + _roi.height <= 0)
        _roi.y = -_roi.height + 1;
    if (_roi.x >= image.cols - 1)
        _roi.x = image.cols - 2;
    if (_roi.y >= image.rows - 1)
        _roi.y = image.rows - 2;

    float cx = _roi.x + _roi.width / 2.0f;//中心点
    float cy = _roi.y + _roi.height / 2.0f;

    float peak_value;
    cv::Point2f res = detect(_tmpl, getFeatures(image, 0, 1.0f), peak_value);//获取response的位置

    float new_peak_value;
    cv::Point2f new_res = detect(_tmpl, getFeatures(image, 0, 1.0f / scale_step), new_peak_value);    // Test at a smaller _scale
    if (0.932 * new_peak_value > peak_value)
    {
        res = new_res;
        peak_value = new_peak_value;
        _scale_x /= scale_step;
        _scale_y /= scale_step;
        _roi.width /= scale_step;
        _roi.height /= scale_step;
        determine(image, cx, cy, res);
        return _roi;
    }
    if (1.02 * new_peak_value > peak_value) //Mark reduce detection times
    {
        determine(image, cx, cy, res);
        return _roi;
    }
    // Test at a bigger _scale
    new_res = detect(_tmpl, getFeatures(image, 0, scale_step), new_peak_value);

    if (scale_weight * new_peak_value < peak_value)
    {
        determine(image, cx, cy, res);
        return _roi;
    }
    res = new_res;
    peak_value = new_peak_value;
    _scale_x *= scale_step;
    _scale_y *= scale_step;
    _roi.width *= scale_step;
    _roi.height *= scale_step;
    determine(image, cx, cy, res);
    return _roi;
}

// Detect object in the current frame.
cv::Point2f KCFTracker::detect(cv::Mat z, cv::Mat x, float &peak_value)
{
    using namespace FFTTools;
    cv::Mat k = gaussianCorrelation(x, z);//作相关运算
    cv::Mat res = (real(fftd(complexMultiplication(_alphaf, k), true)));

    //minMaxLoc only accepts doubles for the peak, and integer points for the coordinates
    cv::Point2i pi;//存放响应response最大值所在的位置
    double pv;//pv存放响应response最大值
    //找到输入数组的最大/最小值，此处寻找最大值，pv存放最大值，pi存放最大值所在的位置
    cv::minMaxLoc(res, NULL, &pv, NULL, &pi);
    peak_value = (float)pv;
    //subpixel peak estimation, coordinates will be non-integer
    cv::Point2f p((float)pi.x, (float)pi.y);

	if (pi.x > 0 && pi.x < res.cols - 1)
    {
        p.x += subPixelPeak(res.at<float>(pi.y, pi.x - 1), peak_value, res.at<float>(pi.y, pi.x + 1));
    }

    if (pi.y > 0 && pi.y < res.rows - 1)
    {
        p.y += subPixelPeak(res.at<float>(pi.y - 1, pi.x), peak_value, res.at<float>(pi.y + 1, pi.x));
    }

    p.x -= (res.cols) / 2;
    p.y -= (res.rows) / 2;

    return p; //responses最大响应对应的目标位置
}

// train tracker with a single image
void KCFTracker::train(cv::Mat x, float train_interp_factor)
{
    using namespace FFTTools;

    cv::Mat k = gaussianCorrelation(x, x);//self-correlation
    cv::Mat alphaf = complexDivision(_prob, (k + lambda)); 

    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor)* x;
    _alphaf = (1 - train_interp_factor) * _alphaf + (train_interp_factor)* alphaf;
}

// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, 
// which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
cv::Mat KCFTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2)
{
    using namespace FFTTools;
    cv::Mat c = cv::Mat(cv::Size(size_patch[1], size_patch[0]), CV_32FC2, cv::Scalar(0, 0)); 
    // HOG features
    if (_hogfeatures)
    {
        cv::Mat caux;
        cv::Mat x1aux;
        cv::Mat x2aux;
        for (int i = 0; i < size_patch[2]; i++)
        {
            x1aux = x1.row(i);   // Procedure do deal with cv::Mat multichannel bug
            x1aux = x1aux.reshape(1, size_patch[0]);
            x2aux = x2.row(i).reshape(1, size_patch[0]);
            cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
            c = c + caux;
        }
    }
    // Gray features
    else
    {
        cv::mulSpectrums(fftd(x1), fftd(x2), c, 0, true);
        c = fftd(c, true);
        rearrange(c);
        c = real(c);
    }
    return c; 
}

// Create Gaussian Peak. Function called only in the first frame.
cv::Mat KCFTracker::createGaussianPeak(int sizey, int sizex)
{
    cv::Mat_<float> res(sizey, sizex);

    int syh = (sizey) / 2;
    int sxh = (sizex) / 2;

    float output_sigma = std::sqrt((float)sizex * sizey) / padding * output_sigma_factor;
    float mult = -0.5 / (output_sigma * output_sigma);

    for (int i = 0; i < sizey; i++)
    {
        for (int j = 0; j < sizex; j++)
        {
            int ih = i - syh;
            int jh = j - sxh;
            res(i, j) = std::exp(mult * (float)(ih * ih + jh * jh));
        }
    }

    return FFTTools::fftd(res);
}

// Obtain sub-window from image, with replication-padding and extract features
cv::Mat KCFTracker::getFeatures(const cv::Mat & image, bool inithann, float scale_adjust)
{
    cv::Rect extracted_roi;

    float cx = _roi.x + _roi.width / 2;
    float cy = _roi.y + _roi.height / 2;//center of area 

    //hanning
    if (inithann)
    {
        int padded_w = _roi.width * padding;
        int padded_h = _roi.height * padding;

        if (padded_w >= padded_h)  //fit to width
        {
            _scale_x = (float)padded_w / (float)template_size;
            _scale_y = (float)padded_h * _scale_x / (float)padded_w;
        }
        else
        {
            _scale_y = (float)padded_h / (float)template_size;
            _scale_x = (float)padded_w * _scale_y / (float)padded_h;
        }
        _tmpl_sz.width = template_size;
        _tmpl_sz.height = template_size;

        if (_hogfeatures)
        {
            // Round to cell size and also make it even
            _tmpl_sz.width = (((int)(_tmpl_sz.width / (2 * cell_size))) * 2 * cell_size) + cell_size * 2;
            _tmpl_sz.height = (((int)(_tmpl_sz.height / (2 * cell_size))) * 2 * cell_size) + cell_size * 2;
        }
        else
        {  //Make number of pixels even (helps with some logic involving half-dimensions)
            _tmpl_sz.width = (_tmpl_sz.width / 2) * 2;
            _tmpl_sz.height = (_tmpl_sz.height / 2) * 2;
        }
    }

    extracted_roi.width = scale_adjust * _scale_x * _tmpl_sz.width;
    extracted_roi.height = scale_adjust * _scale_y * _tmpl_sz.height;

    // center roi with new size
    extracted_roi.x = cx - extracted_roi.width / 2;
    extracted_roi.y = cy - extracted_roi.height / 2;

    cv::Mat FeaturesMap;
    //obtain a subwindow for detection at the position from last
    //	frame, and convert to Fourier domain(its size is unchanged)
    // 在本帧中获取前一帧目标位置的子窗口
    cv::Mat z = RectTools::subwindow(image, extracted_roi, cv::BORDER_REPLICATE);
    //然后要对z提取特征，然后再到频域上与相关滤波器作相关，得到response之后产生新的目标位置
    if (z.cols != _tmpl_sz.width || z.rows != _tmpl_sz.height)
    {
        cv::resize(z, z, _tmpl_sz);
    }

    // HOG features
    if (_hogfeatures)
    {
        IplImage z_ipl = z;
        CvLSVMFeatureMapCaskade *map;
        getFeatureMaps(&z_ipl, cell_size, &map);
        normalizeAndTruncate(map, 0.2f);
        PCAFeatureMaps(map);
        size_patch[0] = map->sizeY;
        size_patch[1] = map->sizeX;
        size_patch[2] = map->numFeatures;

        FeaturesMap = cv::Mat(cv::Size(map->numFeatures, map->sizeX*map->sizeY), CV_32F, map->map);  // Procedure do deal with cv::Mat multichannel bug
        FeaturesMap = FeaturesMap.t();
        freeFeatureMapObject(&map);
    }
    else 
	{ //raw pixel
        FeaturesMap = RectTools::getGrayImage(z);
        FeaturesMap -= (float) 0.5; // In Paper;
        size_patch[0] = z.rows;
        size_patch[1] = z.cols;
        size_patch[2] = 1;
    }

    if (inithann)
    {//只有在第一帧的时候才会用到，创建/初始化 汉宁窗
        createHanningMats();
    }
    FeaturesMap = hann.mul(FeaturesMap);//特征与汉宁窗相乘，起平滑作用
    return FeaturesMap; //最后返回的是与汉宁窗相乘后的结果，，，后续还要进行与相关滤波器作相关
}

// Initialize Hanning window. Function called only in the first frame.
void KCFTracker::createHanningMats()
{
    cv::Mat hann1t = cv::Mat(cv::Size(size_patch[1], 1), CV_32F, cv::Scalar(0));
    cv::Mat hann2t = cv::Mat(cv::Size(1, size_patch[0]), CV_32F, cv::Scalar(0));

    for (int i = 0; i < hann1t.cols; i++)
    {
        hann1t.at<float >(0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
    }
    for (int i = 0; i < hann2t.rows; i++)
    {
        hann2t.at<float >(i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));
    }

    cv::Mat hann2d = hann2t * hann1t;
    // HOG features
    if (_hogfeatures)
    {
        cv::Mat hann1d = hann2d.reshape(1, 1); // Procedure do deal with cv::Mat multichannel bug

        hann = cv::Mat(cv::Size(size_patch[0] * size_patch[1], size_patch[2]), CV_32F, cv::Scalar(0));
        for (int i = 0; i < size_patch[2]; i++)
        {
            for (int j = 0; j < size_patch[0] * size_patch[1]; j++)
            {
                hann.at<float>(i, j) = hann1d.at<float>(0, j);
            }
        }
    }
    // Gray features
    else
    {
        hann = hann2d;
    }
}

// Calculate sub-pixel peak for one dimension
float KCFTracker::subPixelPeak(float left, float center, float right)
{
    float divisor = 2 * center - right - left;
    if (divisor == 0)
        return 0;

    return 0.5 * (right - left) / divisor;
}

