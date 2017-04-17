// MTI805_Laboratoire_2_Panorama.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect/objdetect.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ml;
using namespace std;

//const int GREEN_HUE_HIGH_BLOB = 109;
//const int GREEN_HUE_LOW_BLOB = 20;
//const int GREEN_SAT_HIGH_BLOB = 255;
//const int GREEN_SAT_LOW_BLOB = 33;
//const int GREEN_vOL_HIGH_BLOB = 255;
//const int GREEN_vOL_LOW_BLOB = 0;

const int GREEN_HUE_HIGH_BLOB = 76;
const int GREEN_HUE_LOW_BLOB = 34;
const int GREEN_SAT_HIGH_BLOB = 255;
const int GREEN_SAT_LOW_BLOB = 37;
const int GREEN_vOL_HIGH_BLOB = 255;
const int GREEN_vOL_LOW_BLOB = 0;

const int SIZE_OBJECT_TO_REMOVE_BLOB = 20;

const int WIDTH = 32;
const int HEIGHT = 64;
const int LABEL_POS = 1;
const int LABEL_NEG = -1;

string itos(int i);
bool isGreen(Vec3f color);

int main()
{

	VideoCapture cap("soccer.avi"); // load the video
	String player_cascade_name = "cascade.xml";
	CascadeClassifier player_cascade;
	Mat imgOriginal;
	Mat imgHSV, imgGray;
	Mat imgThresholded_blob;
	Mat imgThresholdedBlob;
	Mat imgMask;
	bool bSuccess = false;
	bool emTrained = false;
	vector<Rect> players;
	vector<Mat> playersImage;
	Mat hist;
	
	// video
	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the mp4 video file." << endl;
		return -1;
	}

	// controls for tuning HSV values
	namedWindow("Control", CV_WINDOW_AUTOSIZE);

	int iLowH = 0;
	int iHighH = 179;

	int iLowS = 0;
	int iHighS = 255;

	int iLowV = 0;
	int iHighV = 255;

	// create trackbars in "Control" window
	cvCreateTrackbar("LowH", "Control", &iLowH, 179); // hue (0 - 179) 
	cvCreateTrackbar("HighH", "Control", &iHighH, 179);

	cvCreateTrackbar("LowS", "Control", &iLowS, 255); // saturation (0 - 255)
	cvCreateTrackbar("HighS", "Control", &iHighS, 255);

	cvCreateTrackbar("LowV", "Control", &iLowV, 255); // value (0 - 255)
	cvCreateTrackbar("HighV", "Control", &iHighV, 255);

	while (cap.isOpened())
	{
		bSuccess = cap.read(imgOriginal); // read a new frame from video		

		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}

		/* Field segmentation using HSV color space */

		cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); // convert the captured frame from BGR to HSV
		//inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded_blob); //Threshold the image
		inRange(imgHSV, Scalar(GREEN_HUE_LOW_BLOB, GREEN_SAT_LOW_BLOB, GREEN_vOL_LOW_BLOB), Scalar(GREEN_HUE_HIGH_BLOB, GREEN_SAT_HIGH_BLOB, GREEN_vOL_HIGH_BLOB), imgThresholded_blob); // Green Hue Value [38;75]

		// morphological opening (remove small objects from the foreground)
		erode(imgThresholded_blob, imgThresholded_blob, getStructuringElement(MORPH_ELLIPSE, Size(SIZE_OBJECT_TO_REMOVE_BLOB, SIZE_OBJECT_TO_REMOVE_BLOB)));
		//dilate(imgThresholded_blob, imgThresholded_blob, getStructuringElement(MORPH_ELLIPSE, Size(15, 15)));

		// morphological closing (fill small holes in the foreground)
		//dilate(imgThresholded_blob, imgThresholded_blob, getStructuringElement(MORPH_ELLIPSE, Size(SIZE_HOLE_TO_FILL, SIZE_HOLE_TO_FILL)));
		//erode(imgThresholded_blob, imgThresholded_blob, getStructuringElement(MORPH_ELLIPSE, Size(SIZE_HOLE_TO_FILL, SIZE_HOLE_TO_FILL)));

		// Set up the detector with default parameters.
		Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create();

		// Detect blobs.
		std::vector<KeyPoint> keypoints;
		detector->detect(imgThresholded_blob, keypoints);
		drawKeypoints(imgThresholded_blob, keypoints, imgThresholded_blob, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

		//vector<Mat> submats;

		//Mat imgMaskBlob(imgOriginal.rows, imgOriginal.cols, CV_32F);
		//vector<Rect> rects;
		//imgMaskBlob.setTo(0);

		//for (int i = 0; i < keypoints.size(); i++) {
		//	float x = keypoints.at(i).pt.x;
		//	float y = keypoints.at(i).pt.y;
		//	float x1 = x - 32 / 2;
		//	float y1 = y - 64 / 2;
		//	float x2 = x + 32 / 2;
		//	float y2 = y + 64 / 2;
		//	Point2f p1(x1, y1);
		//	Point2f p2(x2, y2);
		//	rectangle(imgMaskBlob, p1, p2, Scalar(255, 255, 255), CV_FILLED);
		//	Rect roi(x1, y1, 32, 64);
		//	rects.push_back(roi);
		//	if (roi.width + roi.x >= imgOriginal.cols || roi.height + roi.y >= imgOriginal.rows) {

		//	}
		//	else {
		//		submats.push_back(imgOriginal(roi).clone());
		//	}			
		//}

		//int img_area = WIDTH * HEIGHT;
		//Ptr<SVM> svm = Algorithm::load<SVM>("svm_mti805");
		//float result = 0;
		//string imgname;
		//Mat training_mat(1, img_area, CV_32F);

		//for (int k = 0; k < submats.size(); k++) {
		//	int ii = 0; // Current column in training_mat
		//	for (int i = 0; i < submats.at(k).rows; i++) {
		//		for (int j = 0; j < submats.at(k).cols; j++) {
		//			training_mat.at<int>(0, ii++) = submats.at(k).at<uchar>(i, j);
		//		}
		//	}
		//	result = svm->predict(training_mat);
		//	if (result == LABEL_POS) {
		//		rectangle(imgOriginal, rects.at(k), Scalar(255, 255, 255), 1);
		//	}
		//	else {
		//		rectangle(imgOriginal, rects.at(k), Scalar(0, 0, 0), 1);
		//	}
		//}
		

		// apply mask
		//imgOriginal.copyTo(imgMaskBlob, imgThresholdedBlob);

		// Draw detected blobs as red circles.
		// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
		//Mat im_with_keypoints;
		//drawKeypoints(imgThresholded_blob, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

		// Show blobs
		//imshow("keypoints", im_with_keypoints);
		//imshow("Blob Image", imgMaskBlob);
		imshow("Thresholded Image", imgThresholded_blob); //show the thresholded image
		imshow("Original", imgOriginal); //show the original image
		//imshow("Mask", imgMask); //show the original image

		//waitKey(0);
		if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}

	return 0;
}

string itos(int i) // convert int to string
{
	stringstream s;
	s << i;
	return s.str();
}

bool isGreen(Vec3f color) {
	Mat_<Vec3f> bgr(color);
	//cout << "BGR: " << bgr << endl;
	Mat_<Vec3f> hsv;
	cvtColor(bgr, hsv, CV_BGR2HSV);
	Vec3f hsv_values = hsv.at<Vec3f>(0, 0);
	/*cout << "HSV: " << hsv << endl;
	cout << "hsv_values.val[0] " << hsv_values.val[0] << endl;
	cout << "hsv_values.val[1] " << hsv_values.val[1] << endl;
	cout << "hsv_values.val[2] " << hsv_values.val[2] << endl;*/
	if (hsv_values.val[0] > 30 && hsv_values.val[0] < 150 &&
		hsv_values.val[1] > 15.0f / 100 && hsv_values.val[1] < 75.0f / 100 &&
		hsv_values.val[2] > 0 && hsv_values.val[2] < 255) {
		return true;
	}
	else {
		return false;
	}

}

