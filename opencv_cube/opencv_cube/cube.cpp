#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <iostream>

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

Scalar color(0, 255, 0);
int thickness = 3;

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
			cout << found << endl;
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
	double r2 = u*u + v*v;
	double u_d = (1 + k1*r2 + k2*r2*r2)*u + 2 * p1*u*v + p2*(r2 + 2 * u*u);
	double v_d = (1 + k1*r2 + k2*r2*r2)*v + p1*(r2 + 2 * v*v) + 2 * p2*u*v;
	double x = u_d*fx + cx;
	double y = v_d*fy + cy;
	return Point2f(x, y);
}

int main() {
	cout << "Camera Calibration...." << endl;
	init();
	cout << "Camera Calibration Complete!!" << endl;
	Mat img;
	Size _boardSize(3, 4);
	VideoCapture vc(0);
	if (!vc.isOpened()) return 0;

	bool flag = false;
	cout << "-----------------Start findChessboard Push Space button-----------------" << endl;

	while (1) {		
		Mat rvec, tvec;
		vc >> img;
		Size imageSize = img.size();
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
					objectCorners.push_back(Point3f(i, j, 0.0f));
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
				Mat R;
				Rodrigues(rvec, R);
				Mat R_inv = R.inv();

				//카메라 원점
				Mat P = -R_inv*tvec;
				double* p = (double*)P.data;

				cout << "카메라 원점 : x=" << p[0] << " y=" << p[1] << " z=" << p[2] << endl;

				//pan, tilt
				double z[] = { 0,0,1 };
				Mat Zc(3, 1, CV_64FC1, z);
				Mat Zw = R_inv*Zc;
				double* zw = (double *)Zw.data;
				double pan = atan2(zw[1], zw[0]) - CV_PI / 2.0;
				double tilt = atan2(zw[2], sqrt(zw[0]*zw[0] + zw[1]*zw[1]));
				cout << "Pan : " << pan*180.0/CV_PI << endl;
				cout << "Tilt : " << tilt*180.0/CV_PI << endl;
				cout << endl;

				double p1[] = { 0,0,0 };
				double p2[] = { 0,6,0 };
				double p3[] = { 9,6,0 };
				double p4[] = { 9,0,0 };

				double p5[] = { 0,0,6 };
				double p6[] = { 0,6,6 };
				double p7[] = { 9,6,6 };
				double p8[] = { 9,0,6 };

				line(img, world_to_cam_to_pixel(p1, R, tvec),
					world_to_cam_to_pixel(p2, R, tvec), color, thickness);
				line(img, world_to_cam_to_pixel(p2, R, tvec),
					world_to_cam_to_pixel(p3, R, tvec), color, thickness);
				line(img, world_to_cam_to_pixel(p3, R, tvec),
					world_to_cam_to_pixel(p4, R, tvec), color, thickness);
				line(img, world_to_cam_to_pixel(p4, R, tvec),
					world_to_cam_to_pixel(p1, R, tvec), color, thickness);
				
				line(img, world_to_cam_to_pixel(p5, R, tvec),
					world_to_cam_to_pixel(p6, R, tvec), color, thickness);
				line(img, world_to_cam_to_pixel(p6, R, tvec),
					world_to_cam_to_pixel(p7, R, tvec), color, thickness);
				line(img, world_to_cam_to_pixel(p7, R, tvec),
					world_to_cam_to_pixel(p8, R, tvec), color, thickness);
				line(img, world_to_cam_to_pixel(p8, R, tvec),
					world_to_cam_to_pixel(p5, R, tvec), color, thickness);

				line(img, world_to_cam_to_pixel(p1, R, tvec),
					world_to_cam_to_pixel(p5, R, tvec), color, thickness);
				line(img, world_to_cam_to_pixel(p2, R, tvec),
					world_to_cam_to_pixel(p6, R, tvec), color, thickness);
				line(img, world_to_cam_to_pixel(p3, R, tvec),
					world_to_cam_to_pixel(p7, R, tvec), color, thickness);
				line(img, world_to_cam_to_pixel(p4, R, tvec),
					world_to_cam_to_pixel(p8, R, tvec), color, thickness);				
			}
		}		
		imshow("cam", img);
	}
	destroyAllWindows();

	return 0;
}