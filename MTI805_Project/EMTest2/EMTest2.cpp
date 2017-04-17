#include "stdafx.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::ml;

string itos(int i);
bool isGreen(Vec3f color);
int NB_PLAYERS = 275;
int NB_PARAMS = 3;
int NB_CLUSTERS = 2;
int NB_TEAMS = 3;

int main(int /*argc*/, char** /*argv*/)
{
	bool training = false;
	if (training) {
		Mat hist(NB_PLAYERS, NB_PARAMS, CV_32FC1);
		for (int i = 0; i < NB_PLAYERS; i++) {
			string file = "positives/positive(" + itos(i + 1) + ").jpg";
			Mat source = cv::imread(file);
			//output images
			Mat meanImg(source.rows, source.cols, CV_32FC3);
			Mat fgImg(source.rows, source.cols, CV_8UC3);
			Mat bgImg(source.rows, source.cols, CV_8UC3);

			//convert the input image to float
			Mat floatSource;
			source.convertTo(floatSource, CV_32F);
			Mat source_copy = floatSource.clone();
			//now convert the float image to column vector
			Mat samples(source.rows * source.cols, 3, CV_32FC1);
			int count = 0;
			int idx = 0;

			for (int y = 0; y < source.rows; y++) {
				Vec3f* row = floatSource.ptr<Vec3f>(y);	
				for (int x = 0; x < source.cols; x++) { // rows*cols = 2048
					if (!isGreen(row[x])) { 
						samples.at<Vec3f>(idx, 0) = row[x];
						count++;
						idx++;
						source_copy.at<Vec3f>(y, x)[0] = 0.0f;
						source_copy.at<Vec3f>(y, x)[1] = 0.0f;
						source_copy.at<Vec3f>(y, x)[2] = 0.0f;
					}
				}
			}

			Mat samples_right_size(idx, 3, CV_32FC1);
			//we need just 2 clusters
			for (int y = 0; y < idx; y++) {
				//cout << "samples " << samples.at<Vec3f>(y, 0) << endl;
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
			
			hist.at<float>(i, 0) = r;
			hist.at<float>(i, 1) = g;
			hist.at<float>(i, 2) = b;

			/*cv::imshow("Means", meanImg);
			cv::imshow("Foreground", fgImg);
			cv::imshow("Background", bgImg);*/
			waitKey();
		}

		cout << "Hist : " << hist << endl;

		Ptr<EM> em_model = EM::create();
		em_model->setClustersNumber(NB_TEAMS);
		em_model->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);
		em_model->setTermCriteria(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 300, 0.1));
		em_model->trainEM(hist, noArray(), noArray(), noArray());
		em_model->save("mti805_em");
	}
	else {
		Ptr<EM> em_model = Algorithm::load<EM>("mti805_em");

		for (int i = 1; i < NB_PLAYERS; i++) {
			Mat histSample(1, NB_PARAMS, CV_32FC1);
			string file = "positives/positive(" + itos(i) + ").jpg";
			Mat source = cv::imread(file);

			//convert the input image to float
			Mat floatSource;
			source.convertTo(floatSource, CV_32F);

			//now convert the float image to column vector
			Mat samples(source.rows * source.cols, 3, CV_32FC1);
			int idx = 0;
			for (int y = 0; y < source.rows; y++) {
				cv::Vec3f* row = floatSource.ptr<cv::Vec3f >(y);
				for (int x = 0; x < source.cols; x++) {
					if (!isGreen(row[x])) {
						samples.at<cv::Vec3f >(idx, 0) = row[x];
						idx++;
					}
				}
			}

			Mat samples_right_size(idx, 3, CV_32FC1);
			//we need just 2 clusters
			for (int y = 0; y < idx; y++) {
				//cout << "samples " << samples.at<Vec3f>(y, 0) << endl;
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

			// Using colors independently
			/*histSample.at<float>(0) = ps1[0];
			histSample.at<float>(1) = ps1[1];
			histSample.at<float>(2) = ps1[2];*/
			/*histSample.at<float>(3) = ps1[0];
			histSample.at<float>(4) = ps1[1];
			histSample.at<float>(5) = ps1[2];*/

			// Using luminosity
			/*float luminosity_0 = 0.21 * ps0[0] + 0.72 * ps0[1] + 0.07 * ps0[2];
			float luminosity_1 = 0.21 * ps1[0] + 0.72 * ps1[1] + 0.07 * ps1[2];
			histSample.at<float>(0) = luminosity_0;
			histSample.at<float>(1) = luminosity_1;*/

			//now classify each player
			idx = 0;

			//classify
			int result = cvRound(em_model->predict2(histSample, noArray())[1]);

			if (result == 1) {
				cout << "Positive " << i << " : TEAM A" << endl;
				cout << "histSample " << histSample << endl << endl;
			}
			else if (result == 2) {
				cout << "Positive " << i << " : TEAM B" << endl;
				cout << "histSample " << histSample << endl << endl;
			}
			else {
				cout << "Positive " << i << " : TEAM C" << endl;
				cout << "histSample " << histSample << endl << endl;
			}
			/*cv::imshow("Means", meanImg);
			cv::imshow("Foreground", fgImg);
			cv::imshow("Background", bgImg);*/
			waitKey(0);
		}
	}
	
	system("pause") ;
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
		hsv_values.val[1] > 15.0f/100 && hsv_values.val[1] < 75.0f/100 &&
		hsv_values.val[2] > 0 && hsv_values.val[2] < 255) {
		return true;
	}
	else {
		return false;
	}

}
