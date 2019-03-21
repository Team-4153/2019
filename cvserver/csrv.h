#ifndef _CSRV_H
#define _CSRV_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

struct Camera {
	std::string	name;
	std::string	path;
	int		width;
	int		height;

	// vision target tracking
	bool		targetTrack;
	int		targetHLow, targetSLow, targetVLow;
	int		targetHHigh, targetSHigh, targetVHigh;

	// line tracking
	bool		lineTrack;
	double		lineCoeff;	// used to calculate real distance

	// saving images
	int		savePeriod;	// in ms, 0 == disable

};

struct Stripe {
	cv::Point2f	box[4];
	double		length;
	double		width;
	double		angle;

	void draw(cv::Mat &dst, int idx, int w);
	double centerX();
	double area();
};

struct Target {
	static std::vector<cv::Point3f> model;
	static cv::Mat cameraMatrix;

	Stripe* left;
	Stripe* right;
	cv::Mat	rvec, tvec;
	double	tpos[3];
	double	yaw, pitch, roll;

	Target(Stripe *l, Stripe *r);
	~Target();

	double topWidth();
	double bottomWidth();
	double centerX();
	double width();
	void calcPnP();
};

int csrvInit(void (*frameCallback)(const Camera& c, uint64_t timestamp, cv::Mat frame),
		void (*targetsCallback)(const Camera& c, uint64_t timestamp, Target *center, std::vector<Target *>& targets));

#endif
