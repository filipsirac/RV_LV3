#include <Windows.h>
#include <ctime>

#include <conio.h>
#include <vector>
#include <math.h>

#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>

#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);
VTK_MODULE_INIT(vtkRenderingFreeType);
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyle.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkAxesActor.h>
#include <vtkTextActor.h>
#include <vtkTransform.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkPolyDataNormals.h>
#include <vtkFloatArray.h>
#include <vtkIdList.h>
#include <vtkCallbackCommand.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkSphereSource.h>
#include <vtkCubeSource.h>
#include <vtkCylinderSource.h>
#include <vtkIterativeClosestPointTransform.h>
#include <vtkLandmarkTransform.h>
#include <vtkMatrix4x4.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkPLYReader.h>
#include <vtkPNGWriter.h>
#include <vtkWindowToImageFilter.h>
#include <vtkTextProperty.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>


using namespace cv;
using namespace std;

struct onClickDataClass {
	Mat img;
	int clicks;
};

void onClickCallBack(int event, int x, int y, int flags, void* userData) {
	onClickDataClass* ptrData = (onClickDataClass*)userData;

	if (event == CV_EVENT_LBUTTONDOWN) {
		ptrData->img.at<float>(ptrData->clicks, 0) = x;//x = 0
		ptrData->img.at<float>(ptrData->clicks, 1) = y;//y = 1
		ptrData->clicks++;
	}
}

int main(int argc, char* argv[]) {

	string paramsPath("cameraparams.xml");
	String imgTakenPath("ImgTaken.jbg");

	
	Mat cameraMatrix, distCoeffs;


	FileStorage fsL(paramsPath.c_str(), FileStorage::READ);
	fsL["camera_matrix"] >> cameraMatrix;
	fsL["dist_Coeffs"] >> distCoeffs;
	fsL.release();

	Mat imgTaken;

	VideoCapture cap(0); //Opens a camera for video capturing( 0 open)
	int c = 0;
	if (cap.isOpened()) {

		namedWindow("OriginalView", 1);
		for (;;) {
			c = waitKey(15);

			Mat frame, imgClone, imgUndistored;

			cap >> frame;

			imshow("Original view", frame);

			imgClone = frame.clone();

			undistort(imgClone, imgUndistored, cameraMatrix, distCoeffs);

			if (c == 'p') {
				imgTaken.create(imgUndistored.size(), CV_8UC1);

				cvtColor(imgUndistored, imgTaken, COLOR_BGR2GRAY);

				imshow("Source", imgTaken);
				//save taken image
				imwrite("ImgTaken.jpg",imgTaken);

				break;
			}

			if (c == 27) break; //ESC_KEY = 27
		}

		//imgTaken = imread(imgTakenPath.c_str(), 1);
		int cliks = 0;
		Mat imgPts = Mat::zeros(4, 2, CV_32F);
		Mat objPts = Mat::zeros(4, 3, CV_32FC1);

		onClickDataClass callback;

		callback.img = imgPts;
		callback.clicks = cliks;
		setMouseCallback("Source", onClickCallBack, &callback);
		//imshow("Source", imgTaken);

		while (1) {
			c = waitKey(15);

			if ((c == 27) || (callback.clicks == 4)) {
				setMouseCallback("Source", NULL);
				break;
			}
		}

		//sending x, y, width and height
		Rect rect = Rect((int)(imgPts.at<float>(0, 0)), (int)(imgPts.at<float>(0, 1)),
			(int)(imgPts.at<float>(1, 0) - imgPts.at<float>(0, 0)),
			(int)(imgPts.at<float>(3, 1) - imgPts.at<float>(1, 1)));

		cout << endl << imgTaken.size << endl;

		Mat imgTakenROI = imgTaken(rect);

		imshow("Image ROI test", imgTakenROI);

		//hard coded object points of graph paper
		objPts.at<float>(0, 0) = 0.0f;
		objPts.at<float>(0, 1) = 0.0f;
		objPts.at<float>(0, 2) = 0.0f;

		objPts.at<float>(1, 0) = 1280.0f;
		objPts.at<float>(1, 1) = 0.0f;
		objPts.at<float>(1, 2) = 0.0f;

		objPts.at<float>(2, 0) = 0.0f;
		objPts.at<float>(2, 1) = 720.0f;
		objPts.at<float>(2, 2) = 0.0f;

		objPts.at<float>(3, 0) = 720;
		objPts.at<float>(3, 1) = 1280.0f;
		objPts.at<float>(3, 2) = 0.0f;

		Mat dst, cdst, cdstP;

		//edge detection 
		Canny(imgTakenROI, dst, 50, 200, 3); 

		//copy edges to the images that will display results of the smth
		cvtColor(dst, cdst, COLOR_GRAY2BGR);

		//probabilistic line transformation
		vector<Vec2f> lines;
		HoughLines(dst, lines, 1, CV_PI / 180, 50, 50, 10); 

		//draw the main line
		if (lines.size() > 0) {
			float rho = lines[0][0], theta = lines[0][1];
			Point pt1, pt2;
			double a = cos(theta), d = sin(theta);
			double x0 = a * rho, y0 = d * rho;
			pt1.x = cvRound(x0 + 1000 * (-d));
			pt1.y = cvRound(y0 + 1000 * (a));
			pt2.x = cvRound(x0 - 1000 * (-d));
			pt2.y = cvRound(y0 - 1000 * (a));
			line(cdst, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);

			double cosTheta = a, sinTheta = d;

			rho = rho + imgPts.at<float>(0, 0) * cosTheta + imgPts.at<float>(0, 1) * sinTheta;

			Mat rotationVector, translationVector;
			solvePnP(objPts, imgPts, cameraMatrix, distCoeffs, rotationVector, translationVector);


			// ekstrinzicni parametri
			Mat A = Mat::zeros(3, 3, CV_32FC1);  // cameraMatrix * R
			Mat b = Mat::zeros(3, 1, CV_32FC1); // translacija
			Mat R = Mat::zeros(3, 3, CV_32FC1); // rotacija

			//rotation matrix to rotation vector 
			Rodrigues(rotationVector, R);

			gemm(cameraMatrix, R, 1, 0, 0, A);
			gemm(cameraMatrix, translationVector, 1, 0, 0, b);

			double lambdaX = ((A.at<double>(0, 0) * cosTheta) + (A.at<double>(1, 0) * sinTheta) - (A.at<double>(2, 0) * rho)); 
			double lambdaY = ((A.at<double>(0, 1) * cosTheta) + (A.at<double>(1, 1) * sinTheta) - (A.at<double>(2, 1) * rho));
			double lambdaR = ((b.at<double>(2, 0) * rho) - (b.at<double>(0, 0) * cosTheta) - (b.at<double>(1, 0) * sinTheta));

			double thetaN = atan2(lambdaY, lambdaX);
			double rhoN = lambdaR / sqrt(lambdaX * lambdaX + lambdaY * lambdaY);
			
			char text[7];
			sprintf(text, "%6.2f (mm)", rhoN);
			putText(cdst, text, Point(cdst.size().width / 2, cdst.size().height / 2), FONTDLGORD, 1, CV_RGB(60, 255, 255));
			sprintf(text, "%6.2f (deg)", thetaN * 180 / 3.141592);

			putText(cdst, text, Point(cdst.size().width / 2, cdst.size().height / 2 + 20), FONTDLGORD, 1, CV_RGB(60, 255, 255));
		}

		imshow("Detected lines are red", cdst);
		waitKey();
	}
}