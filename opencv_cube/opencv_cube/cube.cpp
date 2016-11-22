#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <iostream>

using namespace cv;
using namespace std;


Mat cameraMatrix;
Mat distCoeffs;
Size boardSize(4, 4);

vector<vector<Point3f>> objectPoints;
vector<vector<Point2f>> imagePoints;
Size imageSize;
int flag = 0;
vector<Point3f> obj_pts;

void init() {
	for (int i = 0; i < 20; i++) {
		string str = "image" + to_string(i) + ".png";
		Mat img = imread("./calid/" + str);
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
		drawChessboardCorners(img, boardSize, Mat(imageCorners), found);

		if (imageCorners.size() == boardSize.area()) {
			imagePoints.push_back(imageCorners);
			objectPoints.push_back(objectCorners);
		}
	}
	vector<Mat> rvecs, tvecs;
	calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
		distCoeffs, rvecs, tvecs, flag | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5);

	obj_pts.push_back(Point3f(0, 0, 0));
	obj_pts.push_back(Point3f(50, 0, 0));
	obj_pts.push_back(Point3f(0, 50, 0));
	obj_pts.push_back(Point3f(0, 0, 50));
}

int main() {
	cout << "Camera Calibration...." << endl;
	init();
	cout << "Camera Calibration Complete!!" << endl;
	Mat img;
	
	VideoCapture vc(0);
	if (!vc.isOpened()) return 0;

	bool flag = false;
	cout << "Start findChessboard Push Space button!!!!" << endl;

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
			drawChessboardCorners(img, boardSize, Mat(imageCorners), found);
			
			if (imageCorners.size() == boardSize.area()) {
				solvePnP(objectCorners, imageCorners, cameraMatrix, distCoeffs, rvec, tvec);
				Mat R;
				Rodrigues(rvec, R);
				Mat R_inv = R.inv();

				Mat P = -R_inv*tvec;
				double* p = (double*)P.data;

				printf("x=%lf, y=%lf, z=%lf\n", p[0], p[1], p[2]);
			}

			


		}		
		imshow("cam", img);
	}
	destroyAllWindows();

	return 0;
}