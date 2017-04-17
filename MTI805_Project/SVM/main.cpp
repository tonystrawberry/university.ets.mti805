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
const int NUM_POS = 275;
const int NUM_NEG = 826;
const int NUM_BLUE = 123;
const int NUM_WHITE = 129;
const int NUM_YELLOW = 23;
const int NUM_FILES = NUM_BLUE + NUM_WHITE + NUM_YELLOW + NUM_NEG;
const int WIDTH = 32;
const int HEIGHT = 64;
const int LABEL_POS = 2;
const int LABEL_NEG = 4;
const int LABEL_POS_BLUE = 1;
const int LABEL_POS_WHITE = 2;
const int LABEL_POS_YELLOW = 3;
const int NB_TEST_DATA = 20;
const int NB_KEYPOINTS_RETAINED = 10;

int main()
{
	bool training = true;
	bool sift = false;
	if (training && !sift) {
		return training_main();
	}
	else if (training && sift) {
		return training_sift_main();
	} else {
		return test_main();
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

	for (int k = 0; k < NUM_BLUE; k++) {
		string imgname = "pos_blues/blue(" + itos(k + 1) + ").jpg";
		Mat img_mat = imread(imgname, 0); // I used 0 for greyscale
		int ii = 0; // Current column in training_mat
		for (int i = 0; i < img_mat.rows; i++) {
			for (int j = 0; j < img_mat.cols; j++) {
				training_mat.at<int>(k, ii++) = img_mat.at<uchar>(i, j);
			}
		}
		labels.at<int>(k, 0) = LABEL_POS_BLUE;
	}

	for (int k = NUM_BLUE; k < NUM_BLUE + NUM_WHITE; k++) {
		string imgname = "pos_whites/white(" + itos(k + 1) + ").jpg";
		Mat img_mat = imread(imgname, 0); // I used 0 for greyscale
		int ii = 0; // Current column in training_mat
		for (int i = 0; i < img_mat.rows; i++) {
			for (int j = 0; j < img_mat.cols; j++) {
				training_mat.at<int>(k, ii++) = img_mat.at<uchar>(i, j);
			}
		}
		labels.at<int>(k, 0) = LABEL_POS_WHITE;
	}

	for (int k = NUM_BLUE + NUM_WHITE; k < NUM_BLUE + NUM_WHITE + NUM_YELLOW; k++) {
		string imgname = "pos_yellow/yellow(" + itos(k + 1) + ").jpg";
		Mat img_mat = imread(imgname, 0); // I used 0 for greyscale
		int ii = 0; // Current column in training_mat
		for (int i = 0; i < img_mat.rows; i++) {
			for (int j = 0; j < img_mat.cols; j++) {
				training_mat.at<int>(k, ii++) = img_mat.at<uchar>(i, j);
			}
		}
		labels.at<int>(k, 0) = LABEL_POS_YELLOW;
	}

	for (int k = NUM_BLUE + NUM_WHITE + NUM_YELLOW; k < NUM_FILES; k++) {
		string imgname = "negatives/neg(" + itos(k - NUM_POS + 1) + ").jpg";
		Mat img_mat = imread(imgname, 0); // I used 0 for greyscale
		imshow("img", img_mat);
		// waitKey(0);
		int ii = 0; // Current column in training_mat
		cout << k << endl;
		for (int i = 0; i < img_mat.rows; i++) {
			for (int j = 0; j < img_mat.cols; j++) {
				training_mat.at<int>(k, ii++) = img_mat.at<uchar>(i, j);
			}
		}
		labels.at<int>(k, 0) = LABEL_NEG;
	}

	for (int i = 0; i < training_mat.cols; i++) {
		cout << training_mat.at<int>(1100, i) << endl;
	}

	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 200, 1e-6));
	svm->train(training_mat, ROW_SAMPLE, labels);

	svm->save("svm_mti805");
	waitKey(0);
	return 0;
}

int training_sift_main()
{
	int num_files = NUM_FILES;
	Mat training_mat(num_files, NB_KEYPOINTS_RETAINED * 2, CV_32F);
	Mat labels(num_files, 1, CV_32S);

	for (int k = 0; k < NUM_BLUE; k++) {
		string imgname = "pos_blues/blue(" + itos(k + 1) + ").jpg";
		Mat img_mat = imread(imgname, 0); // I used 0 for greyscale
		// Detect key - points / features.
		int minHessian = 400;
		Ptr<SIFT> detector = SIFT::create(minHessian);
		std::vector< KeyPoint > keypoints;
		detector->detect(img_mat, keypoints);

		int ii = 0; // Current column in training_mat
		for (int i = 0; i < NB_KEYPOINTS_RETAINED; i++) {
				training_mat.at<int>(k, ii++) = keypoints.at(i).pt.x;
				training_mat.at<int>(k, ii++) = keypoints.at(i).pt.y;
		}
		labels.at<int>(k, 0) = LABEL_POS_BLUE;
	}

	for (int k = NUM_BLUE; k < NUM_BLUE + NUM_WHITE; k++) {
		string imgname = "pos_whites/white(" + itos(k + 1) + ").jpg";
		Mat img_mat = imread(imgname, 0); // I used 0 for greyscale
		// Detect key - points / features.
		int minHessian = 400;
		Ptr<SIFT> detector = SIFT::create(minHessian);
		std::vector< KeyPoint > keypoints;
		detector->detect(img_mat, keypoints);
		int ii = 0; // Current column in training_mat
		for (int i = 0; i < NB_KEYPOINTS_RETAINED; i++) {
			training_mat.at<int>(k, ii++) = keypoints.at(i).pt.x;
			training_mat.at<int>(k, ii++) = keypoints.at(i).pt.y;
		}
		labels.at<int>(k, 0) = LABEL_POS_WHITE;
	}

	for (int k = NUM_BLUE + NUM_WHITE; k < NUM_BLUE + NUM_WHITE + NUM_YELLOW; k++) {
		string imgname = "pos_yellow/yellow(" + itos(k + 1) + ").jpg";
		Mat img_mat = imread(imgname, 0); // I used 0 for greyscale
		// Detect key - points / features.
		int minHessian = 400;
		Ptr<SIFT> detector = SIFT::create(minHessian);
		std::vector< KeyPoint > keypoints;
		detector->detect(img_mat, keypoints);
		int ii = 0; // Current column in training_mat
		for (int i = 0; i < NB_KEYPOINTS_RETAINED; i++) {
			training_mat.at<int>(k, ii++) = keypoints.at(i).pt.x;
			training_mat.at<int>(k, ii++) = keypoints.at(i).pt.y;
		}
		labels.at<int>(k, 0) = LABEL_POS_YELLOW;
	}

	for (int k = NUM_BLUE + NUM_WHITE + NUM_YELLOW; k < NUM_FILES; k++) {
		string imgname = "negatives/neg(" + itos(k - NUM_POS + 1) + ").jpg";
		Mat img_mat = imread(imgname, 0); // I used 0 for greyscale
		// Detect key - points / features.
		int minHessian = 400;
		Ptr<SIFT> detector = SIFT::create(minHessian);
		std::vector< KeyPoint > keypoints;
		detector->detect(img_mat, keypoints);
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

	svm->save("svm_mti805");
	waitKey(0);
	return 0;
}

int test_main() {
	int img_area = WIDTH * HEIGHT;
	Ptr<SVM> svm = Algorithm::load<SVM>("svm_mti805");
	float result = 0;
	string imgname;
	Mat training_mat(1, img_area, CV_32F);
	for (int k = 0; k < NB_TEST_DATA; k++) {
		imgname = "test/test(" + itos(k + 1) + ").jpg";
		Mat img_mat = imread(imgname, 0); // I used 0 for greyscale
		int ii = 0; // Current column in training_mat
		for (int i = 0; i < img_mat.rows; i++) {
			for (int j = 0; j < img_mat.cols; j++) {
				training_mat.at<int>(0, ii++) = img_mat.at<uchar>(i, j);
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




