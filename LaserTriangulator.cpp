#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core.hpp>
#include "Camera.h"

using namespace cv;
using namespace std;

LaserTriangulator::LaserTriangulator()
{
}

LaserTriangulator::LaserTriangulator(double _focalLength, int _opticalCenterX, int _opticalCenterY, double _laserAngleTheta, double _baseline)
{
	setFocalLength(_focalLength);
	setOpticalCenterX(_opticalCenterX);
	setOpticalCenterY(_opticalCenterY);
	setLaserAngleTheta(_laserAngleTheta);
	setBaseline(_baseline);
	
}

LaserTriangulator::LaserTriangulator(double _focalLength, int _opticalCenterX, int _opticalCenterY, double _laserAngleTheta, double _baseline, double _k1)
{
	setFocalLength(_focalLength);
	setOpticalCenterX(_opticalCenterX);
	setOpticalCenterY(_opticalCenterY);
	setLaserAngleTheta(_laserAngleTheta);
	setBaseline(_baseline);
	setK1(_k1);
}

LaserTriangulator::LaserTriangulator(double _focalLength, int _opticalCenterX, int _opticalCenterY, double _laserAngleTheta, double _baseline, double _k1, double _k2)
{
	setFocalLength(_focalLength);
	setOpticalCenterX(_opticalCenterX);
	setOpticalCenterY(_opticalCenterY);
	setLaserAngleTheta(_laserAngleTheta);
	setBaseline(_baseline);
	setK1(_k1);
	setK2(_k2);
}

LaserTriangulator::LaserTriangulator(double _focalLength, int _opticalCenterX, int _opticalCenterY, double _laserAngleTheta, double _baseline, double _k1, double _k2, double _k3)
{
	setFocalLength(_focalLength);
	setOpticalCenterX(_opticalCenterX);
	setOpticalCenterY(_opticalCenterY);
	setLaserAngleTheta(_laserAngleTheta);
	setBaseline(_baseline);
	setK1(_k1);
	setK2(_k2);
	setK3(_k3);
}

LaserTriangulator::LaserTriangulator(double _focalLength, int _opticalCenterX, int _opticalCenterY, double _laserAngleTheta, double _baseline, double _k1, double _k2, double _k3, double _k4)
{
	setFocalLength(_focalLength);
	setOpticalCenterX(_opticalCenterX);
	setOpticalCenterY(_opticalCenterY);
	setLaserAngleTheta(_laserAngleTheta);
	setBaseline(_baseline);
	setK1(_k1);
	setK2(_k2);
	setK3(_k3);
	setK4(_k4);
}

LaserTriangulator::LaserTriangulator(double _focalLength, int _opticalCenterX, int _opticalCenterY, double _laserAngleTheta, double _baseline, double _k1, double _k2, double _k3, double _k4, double _k5)
{
	setFocalLength(_focalLength);
	setOpticalCenterX(_opticalCenterX);
	setOpticalCenterY(_opticalCenterY);
	setLaserAngleTheta(_laserAngleTheta);
	setBaseline(_baseline);
	setK1(_k1);
	setK2(_k2);
	setK3(_k3);
	setK4(_k4);
	setK5(_k5);
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

void LaserTriangulator::setK1(double _k1)
{
	k1 = _k1;
}

void LaserTriangulator::setK2(double _k2)
{
	k2 = _k2;
}

void LaserTriangulator::setK3(double _k3)
{
	k3 = _k3;
}

void LaserTriangulator::setK4(double _k4)
{
	k4 = _k4;
}

void LaserTriangulator::setK5(double _k5)
{
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

double LaserTriangulator::getK1()
{
	return k1;
}

double LaserTriangulator::getK2()
{
	return k2;
}

double LaserTriangulator::gsetK3()
{
	return k3;
}

double LaserTriangulator::getK4()
{
	return k4;
}

double LaserTriangulator::getK5()
{
	return k5;
}

void LaserTriangulator::calibrateCamera(string path)
{
}

vector<Point3d> LaserTriangulator::pointTriangulation(Mat _binaryImage)
{
	if (_binaryImage.type() != CV_8U)					//Check input format
	{
		cout << "Wrong Mat class type. Prefered type = CV_8U. Please check if image is binary." << endl;
		return;
	}

	vector<Point3d> pointVector;
	Point3f pointCoordinates;

	double focalLength = getFocalLength();
	double baseline = getBaseline();
	int XCameraWorld;
	int YCameraWorld;
	double alphaAngle;
	double betaAngle;
	double thetaAngle = getLaserAngleTheta();
	double c1;

	double x1;
	double y1;
	double z1;

	for (int i = 1; i < _binaryImage.rows - 1; i++)		//Loop doesn't go through whole matrix due to calculation performance 
	{
		for (int j = (_binaryImage.cols / 2 - 0.1 * _binaryImage.cols); j < (_binaryImage.cols / 2 + 0.1 * _binaryImage.cols); j++)
		{
			if (_binaryImage.at<uchar>(i,j) == 255)		//If matrix is of type CV_8U then use Mat.at<uchar>(y,x).
			{
				XCameraWorld = j - getOpticalCenterX();	//Calculate new coordinates with relation to camera optical center
				YCameraWorld = getOpticalCenterY() - i;

				alphaAngle = 90 + atan2(XCameraWorld , focalLength);	//Calculate lengths and angles according to documentation schema
				betaAngle = 180 - alphaAngle - thetaAngle;
				c1 = (sin(thetaAngle) * baseline) / sin(betaAngle);

				z1 = c1 * cos(atan2(XCameraWorld, focalLength));		//Calculate world coordinates
				x1 = (XCameraWorld * z1) / focalLength;
				y1 = (YCameraWorld * z1) / focalLength;

				pointCoordinates.x = x1;								//Save coordinates to Point3f class object
				pointCoordinates.y = y1;
				pointCoordinates.z = z1;

				pointVector.push_back(pointCoordinates);				//Save object to vector containing each point world coordinates
			};
		}
	}
	return pointVector;
}
