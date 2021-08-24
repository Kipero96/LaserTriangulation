#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

enum ThinningTypes
{
	THINNING_ZHANGSUEN_MOD = 0, // Thinning technique of Zhang-Suen (modified - thinning applies only to the vertical screen center)
	THINNING_GUOHALL_MOD = 1  // Thinning technique of Guo-Hall (modified - thinning applies only to the vertical screen center)
};

static void thinningIteration(Mat img, int iter, int thinningType) // Applies a thinning iteration to a binary image
{
	Mat marker = Mat::zeros(img.size(), CV_8UC1);

	if (thinningType == THINNING_ZHANGSUEN_MOD) {
		for (int i = 1; i < img.rows -1 ; i++)
		{
			for (int j = (img.cols/2 - 0.1*img.cols); j < (img.cols / 2 + 0.1 * img.cols); j++)
			{
				uchar p2 = img.at<uchar>(i - 1, j);
				uchar p3 = img.at<uchar>(i - 1, j + 1);
				uchar p4 = img.at<uchar>(i, j + 1);
				uchar p5 = img.at<uchar>(i + 1, j + 1);
				uchar p6 = img.at<uchar>(i + 1, j);
				uchar p7 = img.at<uchar>(i + 1, j - 1);
				uchar p8 = img.at<uchar>(i, j - 1);
				uchar p9 = img.at<uchar>(i - 1, j - 1);

				int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
					(p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
					(p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
					(p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
				int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
				int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
				int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

				if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
					marker.at<uchar>(i, j) = 1;
			}
		}
	}

	if (thinningType == THINNING_GUOHALL_MOD) {
		for (int i = 1; i < img.rows - 1; i++)
		{
			for (int j = (img.cols / 2 - 0.1 * img.cols); j < (img.cols / 2 + 0.1 * img.cols); j++)
			{
				uchar p2 = img.at<uchar>(i - 1, j);
				uchar p3 = img.at<uchar>(i - 1, j + 1);
				uchar p4 = img.at<uchar>(i, j + 1);
				uchar p5 = img.at<uchar>(i + 1, j + 1);
				uchar p6 = img.at<uchar>(i + 1, j);
				uchar p7 = img.at<uchar>(i + 1, j - 1);
				uchar p8 = img.at<uchar>(i, j - 1);
				uchar p9 = img.at<uchar>(i - 1, j - 1);

				int C = ((!p2) & (p3 | p4)) + ((!p4) & (p5 | p6)) +
					((!p6) & (p7 | p8)) + ((!p8) & (p9 | p2));
				int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
				int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
				int N = N1 < N2 ? N1 : N2;
				int m = iter == 0 ? ((p6 | p7 | (!p9)) & p8) : ((p2 | p3 | (!p5)) & p4);

				if ((C == 1) && ((N >= 2) && ((N <= 3)) & (m == 0)))
					marker.at<uchar>(i, j) = 1;
			}
		}
	}

	img &= ~marker;
}

void thinning(InputArray input, OutputArray output, int thinningType) // Apply the thinning procedure to a given image
{
	Mat processed = input.getMat().clone();
	CV_CheckTypeEQ(processed.type(), CV_8UC1, ""); // Enforce the range of the input image to be in between 0 - 255
	processed /= 255;

	Mat prev = Mat::zeros(processed.size(), CV_8UC1);
	Mat diff;

	do {
		thinningIteration(processed, 0, thinningType);
		thinningIteration(processed, 1, thinningType);
		absdiff(processed, prev, diff);
		processed.copyTo(prev);
	} while (countNonZero(diff) > 0);

	processed *= 255;

	output.assign(processed);
}



void main()
{
	VideoCapture cap(0);

	Mat img;
	Mat gray;
	Mat grayBinary;
	Mat imgThinning;

	/*
	///Trackbars for HSV images///

	int hmin = 0, smin = 0, vmin = 230;
	int hmax = 255, smax = 75, vmax = 253;

	namedWindow("Trackbars", (640, 200));
	createTrackbar("Hue Min", "Trackbars", &hmin, 255);
	createTrackbar("Hue Max", "Trackbars", &hmax, 255);
	createTrackbar("Sat Min", "Trackbars", &smin, 255);
	createTrackbar("Sat Max", "Trackbars", &smax, 255);
	createTrackbar("Light Min", "Trackbars", &vmin, 255);
	createTrackbar("Light Max", "Trackbars", &vmax, 255);
	*/

	while (true)
	{
		waitKey(500);					//500ms interval for thinning 
		cap.read(img);					

		cvtColor(img, gray, COLOR_BGR2GRAY);					//convert image captured from camera to greyscale	
		threshold(gray, grayBinary, 252, 255, THRESH_BINARY);	//threshold greyscale image to binary image

		thinning(grayBinary, imgThinning, THINNING_ZHANGSUEN_MOD);	//use modified Zhang-Suen thinning method
		
		imshow("Binary", imgThinning);
	}
}
