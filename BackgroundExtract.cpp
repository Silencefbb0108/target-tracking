#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <bits/stdc++.h>

using namespace std;
using namespace cv;

int main()
{

	Ptr<BackgroundSubtractorMOG2> pBgmodel = createBackgroundSubtractorMOG2();
	pBgmodel->setVarThreshold(20);

	VideoCapture cap;
	cap.open("768x576.avi");
	if( !cap.isOpened() )
	{
		printf("can not open camera or video file\n");
		return -1;
	}


	Mat frame, image, foreGround, backGround, fgMask;

	while (true)
	{
		
		cap >> frame;

		if (frame.empty()) break;

		stringstream ss;
		rectangle(frame, cv::Point(10, 2), cv::Point(100, 20), cv::Scalar(255, 255, 255), -1);
		ss << cap.get(CAP_PROP_POS_FRAMES);
		string frameNumberString = ss.str();
		putText(frame, frameNumberString.c_str(), cv::Point(15, 15), FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

		//缩小为原来四分之一，加快处理速度
		//resize(frame, image, Size(frame.cols / 2, frame.rows / 2), INTER_LINEAR);
		frame.copyTo(image);

		if (foreGround.empty())
			foreGround.create(image.size() , image.type());

		//得到（灰度）前景图像
		pBgmodel->apply(image, fgMask);

		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(fgMask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());

		if( !contours.empty() && !hierarchy.empty() )
		{
			for (int index = 0; index < contours.size(); index++)
			{
			    	if (contourArea(contours[index]) > 500) {
					RotatedRect rectPoint = minAreaRect(contours[index]);
					//cout << contourArea(contours[index]) << endl;
					Point2f fourPoint2f[4];
					rectPoint.points(fourPoint2f);
		 
					for (int i = 0; i < 3; i++)
					{
						line(frame, fourPoint2f[i], fourPoint2f[i + 1], Scalar(0, 0, 255), 0);
					}
					line(frame, fourPoint2f[0], fourPoint2f[3], Scalar(0, 0, 255), 0);
				}
			}
		}
		contours.clear();
		hierarchy.clear();

		GaussianBlur(fgMask, fgMask, Size(5, 5) , 0);
		threshold(fgMask , fgMask, 10, 255, THRESH_BINARY);

		//将foreGraound 所有像素置为0 
		foreGround = Scalar::all(0); 
		//fgMask对应点像素值为255，则foreGround像素值为image对应点像素值，为0则直接为0 
		image.copyTo(foreGround, fgMask);

		pBgmodel->getBackgroundImage(backGround);

		imshow("frame", frame);
		imshow("backGround", backGround);
		imshow("foreGround", foreGround);
		imshow("fgMask", fgMask);

		char key = (char)waitKey(1); 
		if (key == 27) break;

	}

	return 0;

}

