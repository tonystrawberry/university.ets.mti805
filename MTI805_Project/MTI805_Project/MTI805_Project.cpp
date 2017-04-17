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

const int GREEN_HUE_HIGH = 109;
const int GREEN_HUE_LOW = 20;
const int GREEN_SAT_HIGH = 255;
const int GREEN_SAT_LOW = 33;
const int GREEN_vOL_HIGH = 255;
const int GREEN_vOL_LOW = 0;
const int SIZE_HOLE_TO_FILL = 30;
const int SIZE_OBJECT_TO_REMOVE = 20;
const int NB_COLORS = 2;
const int NB_PLAYERS_NEEDED = 60;
const int NB_PARAMS = 3; // corresponding to the three color components
const int NB_TEAMS = 3;
const int MIN_DIST_THRESHOLD = 10;

const int GREEN_HUE_HIGH_BLOB = 76;
const int GREEN_HUE_LOW_BLOB = 34;
const int GREEN_SAT_HIGH_BLOB = 255;
const int GREEN_SAT_LOW_BLOB = 37;
const int GREEN_vOL_HIGH_BLOB = 255;
const int GREEN_vOL_LOW_BLOB = 0;
const int SIZE_OBJECT_TO_REMOVE_BLOB = 10;

bool isGreen(Vec3f color);

int main()
{

	VideoCapture cap("videotest.avi");
	String player_cascade_name = "cascade.xml"; 
	CascadeClassifier player_cascade;

	Mat imgOriginal;
	Mat imgHSV, imgGray;
	Mat imgThresholded;
	Mat imgThresholded_blob;
	Mat imgMask;
	Mat hist;

	bool first = true;
	bool bSuccess = false;
	bool emTrained = false;

	vector<Rect> players;
	vector<Mat> playersImage;
	
	Ptr<EM> em_model = EM::create();
	vector<KeyPoint> keypointsPlayerDetected;
	vector<Rect> playersRectDetected;

	// load the cascade classifier
	if (!player_cascade.load(player_cascade_name)) { 
		std::cout << "Error loading the cascade classifier." << endl;
		return -1; 
	};

	// video
	if (!cap.isOpened())  // if not success, exit program
	{
		std::cout << "Cannot open the mp4 video file." << endl;
		return -1;
	}
	
	// parameters for blob detector
	// setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;

	params.minThreshold = 10;
	params.maxThreshold = 200;
	params.filterByArea = true;
	params.minArea = 100;
	params.filterByCircularity = false;
	params.minCircularity = 0.1;
	params.filterByConvexity = true;
	params.minConvexity = 0.5;
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;

	Ptr<SimpleBlobDetector> blobDetector = SimpleBlobDetector::create(params);

	while(cap.isOpened())
	{		
		bSuccess = cap.read(imgOriginal); // read a new frame from video		

		if (!bSuccess) //if not success, break loop
		{
			std::cout << "Cannot read a frame from video stream" << endl;
			break;
		}

		cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); // convert the captured frame from BGR to HSV

		// field segmentation using HSV color space 			
		inRange(imgHSV, Scalar(GREEN_HUE_LOW, GREEN_SAT_LOW, GREEN_vOL_LOW), Scalar(GREEN_HUE_HIGH, GREEN_SAT_HIGH, GREEN_vOL_HIGH), imgThresholded); // Green Hue Value [38;75]
		// morphological opening (remove small objects from the foreground)
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(SIZE_OBJECT_TO_REMOVE, SIZE_OBJECT_TO_REMOVE)));
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(SIZE_OBJECT_TO_REMOVE, SIZE_OBJECT_TO_REMOVE)));
		// morphological closing (fill small holes in the foreground)
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(SIZE_HOLE_TO_FILL, SIZE_HOLE_TO_FILL)));
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(SIZE_HOLE_TO_FILL, SIZE_HOLE_TO_FILL)));
		// apply mask
		imgOriginal.copyTo(imgMask, imgThresholded);

		// detection of all player's blob
		inRange(imgHSV, Scalar(GREEN_HUE_LOW_BLOB, GREEN_SAT_LOW_BLOB, GREEN_vOL_LOW_BLOB), Scalar(GREEN_HUE_HIGH_BLOB, GREEN_SAT_HIGH_BLOB, GREEN_vOL_HIGH_BLOB), imgThresholded_blob);
		erode(imgThresholded_blob, imgThresholded_blob, getStructuringElement(MORPH_ELLIPSE, Size(SIZE_OBJECT_TO_REMOVE_BLOB, SIZE_OBJECT_TO_REMOVE_BLOB)));
		// detect blobs
		std::vector<KeyPoint> keypoints;
		blobDetector->detect(imgThresholded_blob, keypoints);
		drawKeypoints(imgThresholded_blob, keypoints, imgThresholded_blob, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

		imshow("Thresholded Image", imgThresholded); // show the thresholded image
		imshow("Original", imgOriginal); // show the original image
		imshow("Mask", imgMask); // show the original image

		// player detection 
		cvtColor(imgMask, imgGray, CV_BGR2GRAY); // convert to grayscale
		player_cascade.detectMultiScale(imgGray, players, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30)); 
		Mat imgOriginalClone = imgOriginal.clone();
		for (size_t i = 0; i < players.size(); i++)
		{
			Point center(players[i].x + players[i].width*0.5, players[i].y + players[i].height*0.5);
			ellipse(imgOriginalClone, center, Size(players[i].width*0.5, players[i].height*0.5), 0, 0, 360, Scalar(128, 0, 64), 4, 8, 0);
		}

		// tracking
		// track previous detections
		for (int i = 0; i < keypointsPlayerDetected.size(); i++) {
			float minDist = INFINITY;
			KeyPoint keypointMinDist;
			for (int j = 0; j < keypoints.size(); j++) {				
				if (norm(keypointsPlayerDetected.at(i).pt - keypoints.at(j).pt) < minDist) {
					minDist = norm(keypointsPlayerDetected.at(i).pt - keypoints.at(j).pt);
					keypointMinDist = keypoints.at(j);
				}			
			}
			// distance threshold
			if (minDist < MIN_DIST_THRESHOLD) {
				// update keypointsPlayerDetected
				keypointsPlayerDetected.at(i) = keypointMinDist;
				// update playersRectDetected approximatively
				float x = keypointsPlayerDetected.at(i).pt.x;
				float y = keypointsPlayerDetected.at(i).pt.y;
				float x1 = x - 32 / 2;
				float y1 = y - 64 / 2;
				Point2f p1(x1, y1);
				Rect roi(x1, y1, 32, 64);
				playersRectDetected[i] = roi;
			}
			else {
				// remove because tracked player has disappeared
				keypointsPlayerDetected.erase(keypointsPlayerDetected.begin()+i);
				playersRectDetected.erase(playersRectDetected.begin()+i);
			}
		}

		// initial player detection 
		bool playerPreviousDetected;
		for (int i = 0; i < players.size(); i++) {
			playerPreviousDetected = false;
			int index = -1;
			Rect playerMinDist;
			float minDist = INFINITY;
			for (int j = 0; j < playersRectDetected.size(); j++) {
				Point2f centerPlayerDetected(players[i].x + players[i].width / 2, players[i].y + players[i].height / 2);
				Point2f centerPreviousPlayerDetected(playersRectDetected[j].x + playersRectDetected[j].width / 2, playersRectDetected[j].y + playersRectDetected[j].height / 2);
				if (norm(centerPlayerDetected - centerPreviousPlayerDetected) < minDist) {
					minDist = norm(centerPlayerDetected - centerPreviousPlayerDetected);
					index = j;
				}
			}
			if (minDist < MIN_DIST_THRESHOLD){
				// that player was detected previously
				Point2f centerPlayerDetected(players[i].x + players[i].width / 2, players[i].y + players[i].height / 2);
				Point2f centerPreviousPlayerDetected(playersRectDetected[index].x + playersRectDetected[index].width / 2, playersRectDetected[index].y + playersRectDetected[index].height / 2);
				// draw line between two centers
				line(imgGray, centerPreviousPlayerDetected, centerPlayerDetected, Scalar(0,0,0));
				// update playersRectDetected
				playersRectDetected[index] = players[i];
				playerPreviousDetected = true;
			} 
			if (!playerPreviousDetected) {
				minDist = INFINITY;
				KeyPoint keypointMinDist;
				for (int j = 0; j < keypoints.size(); j++) {
					Point2f centerPlayerDetected(players[i].x + players[i].width / 2, players[i].y + players[i].height / 2);
					if (norm(centerPlayerDetected - keypoints.at(j).pt) < minDist) {
						minDist = norm(centerPlayerDetected - keypoints.at(j).pt);
						keypointMinDist = keypoints.at(j);
					}
				}
				// distance threshold
				if (minDist < MIN_DIST_THRESHOLD) {
					keypointsPlayerDetected.push_back(keypointMinDist);
					playersRectDetected.push_back(players[i]);
				}
			}
		}

		// classification into teams
		Ptr<EM> em_model = Algorithm::load<EM>("mti805_em");
		for (size_t i = 0; i < playersRectDetected.size(); i++)
		{
			Mat histSample(1, NB_PARAMS, CV_32FC1);
			// convert the input image to float
			Mat floatSource;
			Mat imgOriginalCropped = imgOriginal(playersRectDetected[i]);
			imgOriginalCropped.convertTo(floatSource, CV_32F);

			// now convert the float image to column vector
			Mat samples(imgOriginalCropped.rows * imgOriginalCropped.cols, 3, CV_32FC1);
			int idx = 0;
			for (int y = 0; y < imgOriginalCropped.rows; y++) {
				cv::Vec3f* row = floatSource.ptr<cv::Vec3f >(y);
				for (int x = 0; x < imgOriginalCropped.cols; x++) {
					if (!isGreen(row[x])) {
						samples.at<cv::Vec3f >(idx, 0) = row[x];
						idx++;
					}
				}
			}

			Mat samples_right_size(idx, 3, CV_32FC1);
			for (int y = 0; y < idx; y++) {
				samples_right_size.at<Vec3f>(y, 0) = samples.at<Vec3f>(y, 0);
			}

			float total_r = 0.0f;
			float total_g = 0.0f;
			float total_b = 0.0f;
			for (int y = 0; y < idx; y++) {
				total_r += samples_right_size.at<Vec3f>(y, 0)[0];
				total_g += samples_right_size.at<Vec3f>(y, 0)[1];
				total_b += samples_right_size.at<Vec3f>(y, 0)[2];
			}

			float r = total_r / idx;
			float g = total_g / idx;
			float b = total_b / idx;

			histSample.at<float>(0, 0) = r;
			histSample.at<float>(0, 1) = g;
			histSample.at<float>(0, 2) = b;

			// classify
			int result = cvRound(em_model->predict2(histSample, noArray())[1]);

			int width = playersRectDetected[i].width;
			int height = playersRectDetected[i].height;
			if (playersRectDetected[i].x + playersRectDetected[i].width > 854) {
				width -= playersRectDetected[i].x + playersRectDetected[i].width - 854;
			}
			if (playersRectDetected[i].y + playersRectDetected[i].height > 480) {
				height -= playersRectDetected[i].y + playersRectDetected[i].height - 480;
			}
			Rect rect(playersRectDetected[i].x, playersRectDetected[i].y, width, height);

			if (result == 1) {
				rectangle(imgOriginal, rect, Scalar(255, 0, 0), 2, 8, 0);
			}
			else if (result == 2) {
				rectangle(imgOriginal, rect, Scalar(0, 255, 0), 2, 8, 0);
			}
			else {
				rectangle(imgOriginal, rect, Scalar(0, 0, 255), 2, 8, 0);
			}
		}

		imshow("Players detection only", imgOriginalClone);
		imshow("Blob Image", imgThresholded_blob);
		imshow("Players detection with tracking", imgGray);
		imshow("Players detection with tracking", imgOriginal);

		waitKey(0);
 		
		if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			std::cout << "esc key is pressed by user" << endl;
			break;
		}
	}

	return 0;
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

