#include <iostream>
#include <math.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;
using namespace pcl;

class LaserTriangulator
{
private:
	//Every camera has its own intrinsics parameters, therefore you need to calibrate every camera separately
	double focalLength;					//units in pixels
	int opticalCenterX;					//units in pixels
	int opticalCenterY;					//units in pixels
	double k1 = 0.0;					//Barrel radial distortion - typically greater than 0 
	double k2 = 0.0;					//Pincushion radial distortion - typically smaller than 0
	double k3 = 0.0;					//Tangential distortion
	double k4 = 0.0;					//Tangential distortion
	double k5 = 0.0;					//Tangential distortion

	double laserAngleTheta;				//Laser angle to baseline
	double baseline;					//Length between camera axis and laser axis

public:

	LaserTriangulator();
	LaserTriangulator(double _focalLength, int _opticalCenterX, int _opticalCenterY, double _baseline, double _laserAngleTheta = 0.0, double _k1 = 0.0, double _k2 = 0.0, double _k3 = 0.0, double _k4 = 0.0, double _k5 = 0.0);

	~LaserTriangulator();

	void setFocalLength(double _focalLength);
	void setOpticalCenterX(int _opticalCenterX);
	void setOpticalCenterY(int _opticalCenterY);
	void setLaserAngleTheta(double _laserAngleTheta);
	void setBaseline(double _baseline);
	void setK1(double _k1);
	void setK2(double _k2);
	void setK3(double _k3);
	void setK4(double _k4);
	void setK5(double _k5);

	double getFocalLength();
	int getOpticalCenterX();
	int getOpticalCenterY();
	double getLaserAngleTheta();
	double getBaseline();
	double getK1();
	double getK2();
	double gsetK3();
	double getK4();
	double getK5();

	void cameraCalibration(string path, int checkboardWidth, int checkboardHeight, int fieldSize);
	double calculateLaserAngleTheta(double pointX, double pointY, double knownDistance);
	vector<PointXYZ>	pointTriangulation(Mat _binaryImage, int _iterator);
	void thinningIteration(Mat img, int iter, int thinningType);
	void thinning(InputArray input, OutputArray output, int thinningType);
	PointCloud<PointXYZ> generatePointCloud(vector<vector<PointXYZ>>_pointTriangulationVector);

};
