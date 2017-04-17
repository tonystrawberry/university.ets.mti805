// training.cpp : Defines the entry point for the console application.

#include "stdafx.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>

using namespace cv::xfeatures2d;
using namespace std;
using namespace cv;
using namespace cv::ml;

string itos(int i);
int training_main();
int training_sift_main();
int test_main();
int test_main_sift();
const int NUM_POS = 20;
const int NUM_NEG = 20;
const int NUM_FILES = NUM_POS + NUM_NEG;
const int WIDTH = 854;
const int HEIGHT = 480;
const int NB_TEST_DATA = 6;
const int NB_KEYPOINTS_RETAINED = 20;
const int LABEL_POS = 2;
const int LABEL_NEG = -2;

int main()
{
	bool training = false;
	bool sift = true;
	if (training && !sift) {
		return training_main();
	}
	else if (training && sift) {
		return training_sift_main();
	}
	else if (!sift) {
		return test_main();
	}
	else {
		return test_main_sift();
	}
}

string itos(int i) // convert int to string
{
	stringstream s;
	s << i;
	return s.str();
}

int training_main()
{
	int num_files = NUM_FILES;
	int img_area = WIDTH * HEIGHT;
	Mat training_mat(num_files, img_area, CV_32F);
	Mat labels(num_files, 1, CV_32S);


	for (int k = 0; k < NUM_POS; k++) {
		string imgname = "game_views/" + itos(k + 1) + ".png";
		Mat img_mat = imread(imgname, 0); // I used 0 for greyscale
		Mat img_mat_resized(HEIGHT, WIDTH, img_mat.type());
		Size s(HEIGHT, WIDTH);
		resize(img_mat, img_mat_resized, s);
		int ii = 0; // Current column in training_mat
		for (int i = 0; i < img_mat_resized.rows; i++) {
			for (int j = 0; j < img_mat_resized.cols; j++) {
				training_mat.at<int>(k, ii++) = img_mat_resized.at<uchar>(i, j);
			}
		}
		labels.at<int>(k, 0) = LABEL_POS;
	}

	for (int k = NUM_POS; k < NUM_FILES; k++) {
		string imgname = "other_views/" + itos(k - NUM_POS + 1) + ".png";
		Mat img_mat = imread(imgname, 0); // I used 0 for greyscale
		Mat img_mat_resized(HEIGHT, WIDTH, img_mat.type());
		Size s(HEIGHT, WIDTH);
		resize(img_mat, img_mat_resized, s);
		int ii = 0; // Current column in training_mat
		for (int i = 0; i < img_mat_resized.rows; i++) {
			for (int j = 0; j < img_mat_resized.cols; j++) {
				training_mat.at<int>(k, ii++) = img_mat_resized.at<uchar>(i, j);
			}
		}
		labels.at<int>(k, 0) = LABEL_NEG;
	}

	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 200, 1e-6));
	svm->train(training_mat, ROW_SAMPLE, labels);

	svm->save("svm_mti805_view");
	waitKey(0);
	return 0;
}

int training_sift_main()
{
	int num_files = NUM_FILES;
	Mat training_mat(num_files, NB_KEYPOINTS_RETAINED * 2, CV_32F);
	Mat labels(num_files, 1, CV_32S);

	for (int k = 0; k < NUM_POS; k++) {
		string imgname = "game_views/" + itos(k + 1) + ".png";
		Mat img_mat = imread(imgname, 0); // I used 0 for greyscale
		Mat img_mat_resized(HEIGHT, WIDTH, img_mat.type());
		Size s(HEIGHT, WIDTH);
		resize(img_mat, img_mat_resized, s);
		// Detect key - points / features.
		int minHessian = 400;
		Ptr<SIFT> detector = SIFT::create(minHessian);
		std::vector< KeyPoint > keypoints;
		detector->detect(img_mat_resized, keypoints);
		int ii = 0; // Current column in training_mat
		for (int i = 0; i < NB_KEYPOINTS_RETAINED; i++) {
			training_mat.at<int>(k, ii++) = keypoints.at(i).pt.x;
			training_mat.at<int>(k, ii++) = keypoints.at(i).pt.y;
		}
		labels.at<int>(k, 0) = LABEL_POS;
	}

	for (int k = NUM_POS; k < NUM_FILES; k++) {
		string imgname = "other_views/" + itos(k - NUM_POS + 1) + ".png";
		Mat img_mat = imread(imgname, 0); // I used 0 for greyscale
		Mat img_mat_resized(HEIGHT, WIDTH, img_mat.type());
		Size s(HEIGHT, WIDTH);
		resize(img_mat, img_mat_resized, s);
		// Detect key - points / features.
		int minHessian = 400;
		Ptr<SIFT> detector = SIFT::create(minHessian);
		std::vector< KeyPoint > keypoints;
		detector->detect(img_mat_resized, keypoints);
		int ii = 0; // Current column in training_mat
		for (int i = 0; i < NB_KEYPOINTS_RETAINED; i++) {
			training_mat.at<int>(k, ii++) = keypoints.at(i).pt.x;
			training_mat.at<int>(k, ii++) = keypoints.at(i).pt.y;
		}
		labels.at<int>(k, 0) = LABEL_NEG;
	}

	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 200, 1e-6));
	svm->train(training_mat, ROW_SAMPLE, labels);

	svm->save("svm_mti805_view");
	waitKey(0);
	return 0;
}

int test_main() {
	int img_area = WIDTH * HEIGHT;
	Ptr<SVM> svm = Algorithm::load<SVM>("svm_mti805_view");
	float result = 0;
	string imgname;
	Mat training_mat(1, img_area, CV_32F);
	for (int k = 0; k < NB_TEST_DATA; k++) {
		imgname = "test/" + itos(k + 1) + ".png";
		Mat img_mat = imread(imgname, 0); // I used 0 for greyscale
		Mat img_mat_resized(HEIGHT, WIDTH, img_mat.type());
		Size s(HEIGHT, WIDTH);
		resize(img_mat, img_mat_resized, s);
		int ii = 0; // Current column in training_mat
		for (int i = 0; i < img_mat_resized.rows; i++) {
			for (int j = 0; j < img_mat_resized.cols; j++) {
				training_mat.at<int>(0, ii++) = img_mat_resized.at<uchar>(i, j);
			}
		}
		result = svm->predict(training_mat);
		cout << imgname << " " << result << endl;
		imshow("test", img_mat);
		waitKey(0);
	}
	waitKey(0);
	return 0;

}

int test_main_sift() {
	int img_area = WIDTH * HEIGHT;
	Ptr<SVM> svm = Algorithm::load<SVM>("svm_mti805_view");
	float result = 0;
	string imgname;
	Mat training_mat(1, NB_KEYPOINTS_RETAINED * 2, CV_32F);
	for (int k = 0; k < NB_TEST_DATA; k++) {
		imgname = "test/" + itos(k + 1) + ".png";
		Mat img_mat = imread(imgname, 0); // I used 0 for greyscale
		Mat img_mat_resized(HEIGHT, WIDTH, img_mat.type());
		Size s(HEIGHT, WIDTH);
		resize(img_mat, img_mat_resized, s);
		// Detect key - points / features.
		int minHessian = 400;
		Ptr<SIFT> detector = SIFT::create(minHessian);
		std::vector< KeyPoint > keypoints;
		detector->detect(img_mat_resized, keypoints);
		cout << keypoints.size() << endl;
		int ii = 0; // Current column in training_mat
		for (int i = 0; i < NB_KEYPOINTS_RETAINED; i++) {
			training_mat.at<int>(0, ii++) = keypoints.at(i).pt.x;
			training_mat.at<int>(0, ii++) = keypoints.at(i).pt.y;
		}
		result = svm->predict(training_mat);
		cout << imgname << " " << result << endl;
		//imshow("test", img_mat);
		//waitKey(0);
	}
	waitKey(0);
	return 0;

}




