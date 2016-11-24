#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

Mat cameraMatrix;
Mat distCoeffs;

vector<vector<Point3f>> objectPoints;
vector<vector<Point2f>> imagePoints;
Size imageSize;
int flag = 0;

double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0,
k1 = 0.0, k2 = 0.0, p1 = 0.0, p2 = 0.0;

Scalar color(0, 0, 0);
int thickness = 2;

void init() {
	for (int i = 0; i < 20; i++) {
		Size boardSize(7, 4);
		string str = "image" + to_string(i) + ".png";
		Mat img = imread("./calib/" + str);
		imageSize = img.size();

		vector<Point3f> objectCorners;
		vector<Point2f> imageCorners;

		Mat img_gray;
		for (int i = 0; i < boardSize.height; i++) {
			for (int j = 0; j < boardSize.width; j++) {
				objectCorners.push_back(Point3f(i, j, 0.0f));
			}
		}

		cvtColor(img, img_gray, COLOR_BGR2GRAY);
		
		bool found = findChessboardCorners(img, boardSize, imageCorners);
		if (found)
			cornerSubPix(img_gray, imageCorners, Size(11, 11), Size(-1, -1),
				TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		else
			std::cout << found << endl;
		drawChessboardCorners(img, boardSize, Mat(imageCorners), found);

		if (imageCorners.size() == boardSize.area()) {			
			imagePoints.push_back(imageCorners);
			objectPoints.push_back(objectCorners);
		}
	}
	vector<Mat> rvecs, tvecs;
	calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
		distCoeffs, rvecs, tvecs, flag | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5);

	fx = cameraMatrix.at<double>(0, 0);
	cx = cameraMatrix.at<double>(0, 2);
	fy = cameraMatrix.at<double>(1, 1);
	cy = cameraMatrix.at<double>(1, 2);
	k1 = distCoeffs.at<double>(0, 0);
	k2 = distCoeffs.at<double>(0, 1);
	p1 = distCoeffs.at<double>(0, 2);
	p2 = distCoeffs.at<double>(0, 3);
}

Point2f world_to_cam_to_pixel(double w[], Mat R, Mat tvec) {
	Mat Pw(3, 1, CV_64FC1, w);
	Mat Pc = R*Pw + tvec;
	double* pc = (double*)Pc.data;
	double u = pc[0] / pc[2];
	double v = pc[1] / pc[2];
	double x = u*fx + cx;
	double y = v*fy + cy;
	return Point2f(x, y);
}

double _p1[] = { 0,0,0 };
double _p2[] = { 0,8.5,0 };
double _p3[] = { 10,8.5,0 };
double _p4[] = { 10,0,0 };

double _p5[] = { 0,0,3.3 };
double _p6[] = { 0,8.5,3.3 };
double _p7[] = { 10,8.5,3.3 };
double _p8[] = { 10,0,3.3 };

Mat R;
Mat rvec, tvec;

Point2f _p1_p;
Point2f _p2_p;
Point2f _p3_p;
Point2f _p4_p;
Point2f _p5_p;
Point2f _p6_p;
Point2f _p7_p;
Point2f _p8_p;

int main() {
	std::cout << "Camera Calibration...." << endl;
	init();
	std::cout << "Camera Calibration Complete!!" << endl;
	Mat img;
	Size _boardSize(3, 4);
	VideoCapture vc(0);
	if (!vc.isOpened()) return 0;

	bool flag = false;
	std::cout << "-----------------Start findChessboard Push Space button-----------------" << endl;
	std::cout << "Press Space........." << endl;

	while (1) {
		vc >> img;
		if (img.empty()) break;		
		
		int keycode = waitKey(33);
		if (keycode == 27) { //esc
			break;
		}
		if (keycode == 32 && flag == false) {
			flag = true;
		}
		else if (keycode == 32 && flag == true) {
			flag = false;
		}
		if (flag == true) {
			vector<Point3f> objectCorners;
			vector<Point2f> imageCorners;
			Mat img_gray;
			for (int i = 0; i < _boardSize.height; i++) {
				for (int j = 0; j < _boardSize.width; j++) {
					objectCorners.push_back(Point3f(i*3, j*3, 0.0f));
				}
			}

			cvtColor(img, img_gray, COLOR_BGR2GRAY);

			bool found = findChessboardCorners(img, _boardSize, imageCorners);
			if (found)
				cornerSubPix(img_gray, imageCorners, Size(11, 11), Size(-1, -1),
					TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			//drawChessboardCorners(img, _boardSize, Mat(imageCorners), found);

			if (imageCorners.size() == _boardSize.area()) {
				Scalar pointcolor(219, 59, 38);
				circle(img, imageCorners.at(0), 3, pointcolor, thickness);
				circle(img, imageCorners.at(1), 3, pointcolor, thickness);
				circle(img, imageCorners.at(2), 3, pointcolor, thickness);
				circle(img, imageCorners.at(5), 3, pointcolor, thickness);
				circle(img, imageCorners.at(8), 3, pointcolor, thickness);
				circle(img, imageCorners.at(11), 3, pointcolor, thickness);
				circle(img, imageCorners.at(10), 3, pointcolor, thickness);
				circle(img, imageCorners.at(9), 3, pointcolor, thickness);
				circle(img, imageCorners.at(6), 3, pointcolor, thickness);
				circle(img, imageCorners.at(3), 3, pointcolor, thickness);
				
				solvePnP(objectCorners, imageCorners, cameraMatrix, distCoeffs, rvec, tvec);
				
				Rodrigues(rvec, R);
				Mat R_inv = R.inv();

				//카메라 원점
				Mat P = -R_inv*tvec;
				double* p = (double*)P.data;

				std::cout << "x=" << p[0] << " y=" << p[1] << " z=" << p[2] << endl;

				//pan, tilt
				double z[] = { 0,0,1 };
				Mat Zc(3, 1, CV_64FC1, z);
				Mat Zw = R_inv*Zc;
				double* zw = (double *)Zw.data;
				double pan = atan2(zw[1], zw[0]) - CV_PI / 2.0;
				double tilt = atan2(zw[2], sqrt(zw[0]*zw[0] + zw[1]*zw[1]));
				std::cout << "Pan: " << pan*180.0/CV_PI << endl;
				std::cout << "Tilt: " << tilt*180.0/CV_PI << endl;
				std::cout << endl;
			}
		}

		if (!R.empty() && !tvec.empty() && !rvec.empty()) {
			_p1_p = world_to_cam_to_pixel(_p1, R, tvec);
			_p2_p = world_to_cam_to_pixel(_p2, R, tvec);
			_p3_p = world_to_cam_to_pixel(_p3, R, tvec);
			_p4_p = world_to_cam_to_pixel(_p4, R, tvec);
			_p5_p = world_to_cam_to_pixel(_p5, R, tvec);
			_p6_p = world_to_cam_to_pixel(_p6, R, tvec);
			_p7_p = world_to_cam_to_pixel(_p7, R, tvec);
			_p8_p = world_to_cam_to_pixel(_p8, R, tvec);

			cv::line(img, _p1_p, _p2_p, color, thickness);
			cv::line(img, _p2_p, _p3_p, color, thickness);
			cv::line(img, _p3_p, _p4_p, color, thickness);
			cv::line(img, _p4_p, _p1_p, color, thickness);

			cv::line(img, _p5_p, _p6_p, color, thickness);
			cv::line(img, _p6_p, _p7_p, color, thickness);
			cv::line(img, _p7_p, _p8_p, color, thickness);
			cv::line(img, _p8_p, _p5_p, color, thickness);

			cv::line(img, _p1_p, _p5_p, color, thickness);
			cv::line(img, _p2_p, _p6_p, color, thickness);
			cv::line(img, _p3_p, _p7_p, color, thickness);
			cv::line(img, _p4_p, _p8_p, color, thickness);

			Scalar textcolor(255, 255, 255);
			cv::putText(img, "(0, 0, 0)", _p1_p, FONT_HERSHEY_SIMPLEX, 0.5, textcolor, 1);
			cv::putText(img, "(0, 8.5, 0)", _p2_p, FONT_HERSHEY_SIMPLEX, 0.5, textcolor, 1);
			cv::putText(img, "(10, 8.5, 0)", _p3_p, FONT_HERSHEY_SIMPLEX, 0.5, textcolor, 1);
			cv::putText(img, "(10, 0, 0)", _p4_p, FONT_HERSHEY_SIMPLEX, 0.5, textcolor, 1);
			cv::putText(img, "(0, 0, 3.3)", _p5_p, FONT_HERSHEY_SIMPLEX, 0.5, textcolor, 1);
			cv::putText(img, "(0, 8.5, 3.3)", _p6_p, FONT_HERSHEY_SIMPLEX, 0.5, textcolor, 1);
			cv::putText(img, "(10, 8.5, 3.3)", _p7_p, FONT_HERSHEY_SIMPLEX, 0.5, textcolor, 1);
			cv::putText(img, "(10, 0, 3.3)", _p8_p, FONT_HERSHEY_SIMPLEX, 0.5, textcolor, 1);
		}
		imshow("cam", img);
	}
	destroyAllWindows();

	return 0;
}