// KCF_Tracking.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "kcftracker.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <ctime>

using namespace std;
using namespace cv;

int main() {
	
	Rect2d roi;
	Mat frame;

	KCFTracker tracker(true);

	//std::string video = "768x576.avi";
	//VideoCapture cap(video);
	VideoCapture cap(0);

	cap >> frame;
	
	roi = selectROI("tracker", frame, true, false);
	cout << roi.x << " " << roi.y << " " << roi.width << " " << roi.height << endl;

	if (roi.width == 0 || roi.height == 0) return 0;

	tracker.init(roi, frame);

	printf("Start the tracking process\n");
	clock_t start = clock();

	for (;; ) {

		cap >> frame;
		//cap.read(frame);

		if (frame.rows == 0 || frame.cols == 0) break;

		roi = tracker.update(frame);

		rectangle(frame, roi, Scalar(0, 0, 255), 1, 8);

		imshow("tracker", frame);

		if (waitKey(1) == 27) break;
	}

	clock_t finish = clock();
	cout << (finish - start) / CLOCKS_PER_SEC << endl;

	return 0;
}



