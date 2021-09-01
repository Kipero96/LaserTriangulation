#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS
#include "LaserTriangulator.h"

using namespace cv;
using namespace std;

LaserTriangulator::LaserTriangulator(double _focalLength, int _opticalCenterX, int _opticalCenterY, double _baseline, double _laserAngleTheta,
									 double _k1, double _k2, double _k3, double _k4, double _k5)
{
	setFocalLength(_focalLength);
	setOpticalCenterX(_opticalCenterX);
	setOpticalCenterY(_opticalCenterY);
	setBaseline(_baseline);
	setLaserAngleTheta(_laserAngleTheta);
	setK(double _k1, double _k2, double _k3, double _k4, double _k5);
}

LaserTriangulator::~LaserTriangulator()
{
	cout << "Object deleted" << endl;
}

void LaserTriangulator::setFocalLength(double _focalLength)
{
	focalLength = _focalLength;
}

void LaserTriangulator::setOpticalCenterX(int _opticalCenterX)
{
	opticalCenterX = _opticalCenterX;
}

void LaserTriangulator::setOpticalCenterY(int _opticalCenterY)
{
	opticalCenterY = _opticalCenterY;
}

void LaserTriangulator::setLaserAngleTheta(double _laserAngleTheta)
{
	laserAngleTheta = _laserAngleTheta;
}

void LaserTriangulator::setBaseline(double _baseline)
{
	baseline = _baseline;
}
void LaserTriangulator::setK(double _k1, double _k2, double _k3, double _k4, double _k5)
{
	k1 = _k1;
	k2 = _k2;
	k3 = _k3;
	k4 = _k4;
	k5 = _k5;
}

double LaserTriangulator::getFocalLength()
{
	return focalLength;
}

int LaserTriangulator::getOpticalCenterX()
{
	return opticalCenterX;
}

int LaserTriangulator::getOpticalCenterY()
{
	return opticalCenterY;
}

double LaserTriangulator::getLaserAngleTheta()
{
	return laserAngleTheta;
}

double LaserTriangulator::getBaseline()
{
	return baseline;
}

void LaserTriangulator::cameraCalibration(string path, int checkboardWidth, int checkboardHeight, int fieldSize)
{
	string fileExtension = "/Image*.png";				//Create file extension for file path
	string filePath = path + fileExtension;

	vector<String> fileNames;

	glob(filePath, fileNames, false);
	Size patternSize(checkboardWidth - 1, checkboardHeight - 1);
	vector<vector<Point2f>> q(fileNames.size());

	vector<vector<Point3f>> Q;
	// 1. Generate checkerboard (world) coordinates Q. The board has 25 x 18
	// fields with a size of 15x15mm

	int checkerBoard[2] = { checkboardWidth, checkboardHeight };
	// Defining the world coordinates for 3D points
	vector<Point3f> objp;
	for (int i = 1; i < checkerBoard[1]; i++) {
		for (int j = 1; j < checkerBoard[0]; j++) {
			objp.push_back(Point3f(j * fieldSize, i * fieldSize, 0));
		}
	}

	vector<Point2f> imgPoint;
	// Detect feature points
	size_t i = 0;
	for (auto const& f : fileNames) {
		cout << string(f) << endl;

		// 2. Read in the image an call findChessboardCorners()
		Mat img = imread(fileNames[i]);
		Mat gray;

		cvtColor(img, gray, COLOR_RGB2GRAY);

		bool patternFound = findChessboardCorners(gray, patternSize, q[i], CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);

		// 2. Use cornerSubPix() to refine the found corner detections
		if (patternFound) {
			cornerSubPix(gray, q[i], Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.1));
			Q.push_back(objp);
		}

		// Display
		drawChessboardCorners(img, patternSize, q[i], patternFound);
		imshow("chessboard detection", img);
		waitKey(0);

		i++;
	}


	Matx33d cameraMatrix(Matx33d::eye());
	Vec<double, 5> distorionCoefficients(0, 0, 0, 0, 0);
	Mat rotationMatrix, translationMatrix;



	int flags = CALIB_FIX_ASPECT_RATIO + CALIB_FIX_K3 + CALIB_ZERO_TANGENT_DIST + CALIB_FIX_PRINCIPAL_POINT;
	Size frameSize(1920, 1080);

	cout << "Calibrating..." << endl;

	float error = calibrateCamera(Q, q, frameSize, cameraMatrix, distorionCoefficients, rotationMatrix, translationMatrix, flags);

	setFocalLength(distorionCoefficients(1, 1));
	setOpticalCenterX(distorionCoefficients(1, 3));
	setOpticalCenterY(distorionCoefficients(2, 3));
	setK(distorionCoefficients(1), distorionCoefficients(2), distorionCoefficients(3), distorionCoefficients(4), distorionCoefficientsv(5));

	cout << "Reprojection error: " << error << "\nIntrinsic matrix: \n" << cameraMatrix << "\nDistortion vector: \n" << distorionCoefficients << endl;
	cout << "Rotation matrix for each frame: " << rotationMatrix << endl;
	cout << "Translation matrix for each frame: " << translationMatrix << endl;
}

double LaserTriangulator::calculateLaserAngleTheta(double pointX, double pointY, double knownDistance)
{
	double XCameraWorld1;
	double YCameraWorld1;

	XCameraWorld1 = pointX - getOpticalCenterX();	//Calculate new coordinates with relation to camera optical center
	YCameraWorld1 = getOpticalCenterY() - pointY;

	double alphaAngle = M_PI_2 + atan2(XCameraWorld1, getFocalLength());
	double x0 = (XCameraWorld1 * knownDistance) / getFocalLength();
	double c0 = sqrt(pow(x0, 2) + pow(knownDistance, 2));
	double a0 = sqrt(pow(getBaseline(), 2) + pow(c0, 2) - 2 * getBaseline() * c0 * cos(alphaAngle));

	double thetaAngle = asin((sin(alphaAngle) * c0) / a0);
	setLaserAngleTheta(thetaAngle);

	cout << "Laser angle set to: " << thetaAngle * (180 / M_PI) << endl;

	return thetaAngle;
}

vector<PointXYZ> LaserTriangulator::pointTriangulation(Mat _binaryImage, int _iterator)
{
	if (_binaryImage.type() != CV_8U)					//Check input format
	{
		cout << "Wrong Mat class type. Prefered type = CV_8U. Please check if image is binary." << endl;
	}

	vector<PointXYZ> pointVector;
	PointXYZ pointCoordinates;

	double focalLength = getFocalLength() / 3;
	double baseline = getBaseline();
	int XCameraWorld;
	int YCameraWorld;
	double alphaAngle;
	double betaAngle;
	double thetaAngle = getLaserAngleTheta();
	double c1;

	float x1;
	float y1;
	float z1;

	for (int i = 1; i < _binaryImage.rows - 1; i++)		//Loop doesn't go through whole matrix due to calculation performance 
	{
		for (int j = 1; j < _binaryImage.cols / 2; j++)
		{
			if (_binaryImage.at<uchar>(i, j) == 255)		//If matrix is of type CV_8U then use Mat.at<uchar>(y,x).
			{
				XCameraWorld = j - getOpticalCenterX() / 3;	//Calculate new coordinates with relation to camera optical center
				YCameraWorld = getOpticalCenterY() / 3 - i;

				alphaAngle = M_PI_2 + atan2(XCameraWorld, focalLength);	//Calculate lengths and angles according to documentation schema
				betaAngle = M_PI - alphaAngle - thetaAngle;
				c1 = (sin(thetaAngle) * baseline) / sin(betaAngle);

				z1 = c1 * cos(atan2(XCameraWorld, focalLength));		//Calculate world coordinates
				x1 = (XCameraWorld * z1) / focalLength;
				y1 = (YCameraWorld * z1) / focalLength;

				pointCoordinates.x = x1+_iterator;								//Save coordinates to PointXYZ class object
				pointCoordinates.y = y1;
				pointCoordinates.z = z1;

				pointVector.push_back(pointCoordinates);				//Save object to vector containing each point world coordinates
			};
		}
	}
	return pointVector;
}

enum ThinningTypes
{
	THINNING_ZHANGSUEN_MOD = 0, // Thinning technique of Zhang-Suen (modified - thinning applies only to the vertical screen center)
	THINNING_GUOHALL_MOD = 1  // Thinning technique of Guo-Hall (modified - thinning applies only to the vertical screen center)
};

void LaserTriangulator::thinningIteration(Mat img, int iter, int thinningType) // Applies a thinning iteration to a binary image
{
	Mat marker = Mat::zeros(img.size(), CV_8UC1);

	if (thinningType == THINNING_ZHANGSUEN_MOD) {
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

void LaserTriangulator::thinning(InputArray input, OutputArray output, int thinningType) // Apply the thinning procedure to a given image
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

PointCloud<PointXYZ> LaserTriangulator::generatePointCloud(vector<vector<PointXYZ>> _pointTriangulationVector)
{
	PointCloud<PointXYZ> cloud;

	for (int i = 0; i <= _pointTriangulationVector.size(); i++)
	{
		for (int j = 0; j <= _pointTriangulationVector[i].size(); j++)
		{
			cloud.push_back(PointXYZ(_pointTriangulationVector[i][j].x, _pointTriangulationVector[i][j].y, _pointTriangulationVector[i][j].z));
		}
	}

	return cloud;
}
