/*----------------------------------------------------------------------------*/
/* Copyright (c) 2018 FIRST. All Rights Reserved.                             */
/* Open Source Software - may be modified and shared by FRC teams. The code   */
/* must be accompanied by the FIRST BSD license file in the root directory of */
/* the project.                                                               */
/*----------------------------------------------------------------------------*/

#include <cstdio>
#include <string>
#include <vector>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
//#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d/calib3d_c.h>
#include <networktables/NetworkTableInstance.h>
#include <wpi/StringRef.h>
#include <wpi/json.h>
#include <wpi/raw_istream.h>
#include <wpi/raw_ostream.h>
#include <cscore/cscore_oo.h>

#include "cameraserver/CameraServer.h"

// Scale factor. Should be 1 for the real robot
#define SCALE 1

// Set to 0 for not printing anything on the screen
#define DEBUG 0

/*
   JSON format:
   {
       "team": <team number>,
       "ntmode": <"client" or "server", "client" if unspecified>
       "cameras": [
           {
               "name": <camera name>
               "path": <path, e.g. "/dev/video0">
               "pixel format": <"MJPEG", "YUYV", etc>   // optional
               "width": <video mode width>              // optional
               "height": <video mode height>            // optional
               "fps": <video mode fps>                  // optional
               "brightness": <percentage brightness>    // optional
               "white balance": <"auto", "hold", value> // optional
               "exposure": <"auto", "hold", value>      // optional
               "properties": [                          // optional
                   {
                       "name": <property name>
                       "value": <property value>
                   }
               ]
           }
       ]
   }
 */

using namespace cv;
using namespace std;
//using namespace cv::xfeatures2d;

static const char* configFile = "/boot/frc.json";
unsigned int team;
bool server = false;

struct CameraConfig {
	string		name;
	string		path;
	int		width;
	int		height;

	// vision target tracking
	bool		targetTrack;
	int		targetHLow, targetSLow, targetVLow;
	int		targetHHigh, targetSHigh, targetVHigh;

	// line tracking
	bool		lineTrack;
	double		lineCoeff;	// used to calculate real distance

	wpi::json	config;
};

vector<CameraConfig> cameras;

nt::NetworkTableInstance ntinst;
std::shared_ptr<nt::NetworkTable> target_table, line_table;

Mat erosionElement;

struct Stripe {
	Point2f	box[4];
	double	length;
	double	width;
	double	angle;

//	bool left() { return angle < 0.0; }
	void draw(Mat &dst, int idx, int w) {
		Scalar color[] = { Scalar(0, 0, 255), Scalar(255, 0, 0), Scalar(255, 255, 255) };
//		Scalar color = left?Scalar(0, 0, 255):Scalar(255, 0, 0);

		line(dst, box[0], box[1], color[idx], w);
		line(dst, box[1], box[3], color[idx], w);
		line(dst, box[3], box[2], color[idx], w);
		line(dst, box[2], box[0], color[idx], w);
	}

	double area() {
		return length * width;
	}
};

struct Target {
	Stripe* left;
	Stripe* right;
//	Mat	h;
	Mat	rvec, tvec;
	double	tpos[3];
	double	yaw, pitch, roll;
	static vector<Point3f> model;
	static Mat cameraMatrix;

	Target(Stripe *l, Stripe *r) {
		left = l;
		right = r;
		double avg = (topWidth() + bottomWidth())/2.0;
		calcPnP();
	}

	~Target() {
		delete(left);
		delete(right);
	}

	double topWidth() { return right->box[1].x - left->box[0].x; }
	double bottomWidth() { return right->box[3].x - left->box[2].x; }
	void calcPnP();
};

const int width = 640; // 1640 // 1280 // 640
const int height = 480; // 922 // 720 // 480

// stripe data
const double stripe_ratio = 2.78;
const double stripe_err = 2.0;
const double angle_low = -20.0;
const double angle_high = 20.0;

// Model of the target. All numbers are in centimeters
vector<Point3f> Target::model = {
	// left stripe
	Point3f(-15.3967/SCALE, 99.3775/SCALE, 0),
	Point3f(-10.16/SCALE, 98.1442/SCALE, 0),
	Point3f(-18.8945/SCALE, 85.8525/SCALE, 0),
	Point3f(-13.6578/SCALE, 84.6192/SCALE, 0),

	// right stripe
	Point3f(15.3967/SCALE, 99.3775/SCALE, 0),
	Point3f(10.16/SCALE, 98.1442/SCALE, 0),
	Point3f(18.8945/SCALE, 85.8525/SCALE, 0),
	Point3f(13.6578/SCALE, 84.6192/SCALE, 0),
};

// Raspberry Pi Camera (640x480)
//Mat Target::cameraMatrix = (Mat_<double>(3,3) <<  4.6676194913596777e+02, 0., 3.1086529034645207e+02, 0., 4.6676194913596777e+02, 2.3946292934201807e+02, 0., 0., 1.);

// Microsoft HD Camera (640x480)
Mat Target::cameraMatrix = (Mat_<double>(3,3) <<  6.7939614509180524e+02, 0., 3.0627626279973128e+02, 0., 6.7939614509180524e+02, 2.2394196729643429e+02, 0., 0., 1.);

void Target::calcPnP() {
	int i;
	vector<Point2f> pts;

	for(i = 0; i < 4; i++)
		pts.push_back(left->box[i]);

	for(i = 0; i < 4; i++)
		pts.push_back(right->box[i]);

	Mat d;
	bool ret = solvePnP(model, pts, cameraMatrix, d, rvec, tvec);

	double *t = (double *) tvec.datastart;
	tpos[0] = t[0];
	tpos[1] = t[1];
	tpos[2] = t[2];

	Mat rvec1;
	Rodrigues(rvec, rvec1);

	Vec3d eulerAngles;
	Mat cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ;
	double* r = rvec1.ptr<double>();
	double projMatrix[12] = {
		r[0], r[1], r[2], 0,
                r[3], r[4], r[5], 0,
                r[6], r[7], r[8], 0
	};

	decomposeProjectionMatrix( Mat(3,4,CV_64FC1, projMatrix), cameraMatrix,
		rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles);

	yaw   = eulerAngles[1]; 
	pitch = eulerAngles[0];
	roll  = eulerAngles[2];
}

double distance(Point2f& p1, Point2f& p2) {
	double dx, dy;

	dx = p1.x - p2.x;
	dy = p1.y - p2.y;

	return sqrt(dx*dx + dy*dy);
}

int cmppoint(const void *p1, const void *p2) {
	int ret;

	Point *pt1 = (Point *) p1;
	Point *pt2 = (Point *) p2;

	ret = pt1->y - pt2->y;
	if (ret == 0)
		ret = pt1->x - pt2->x;

	return ret;
}

double dist2(Point2f& p1, Point2f& p2) {
	double dx = p1.x - p2.x;
	double dy = p1.y - p2.y;

	return dx*dx + dy*dy;
}

bool checkStripe(Stripe *s) {
	double r = s->length / s->width;

	// check if the stripe is big enough
	if (s->area() < 36) {
//		printf("\tbad area size: %f\n", s->area());
		return false;
	}

	// check if the length and the width are within expected ratio range
	if (isnan(r) || r < (stripe_ratio - stripe_err) || r > (stripe_ratio + stripe_err)) {
//		printf("\tbad stripe ratio: %f\n", r);
		return false;
	}

	// check if the angle of the stripe is within the expected range
	if (s->angle < angle_low || s->angle > angle_high) {
//		printf("\tbad angle: %f should be between (%f, %f)\n", s->angle, angle_low, angle_high);
		return false;
	}

	return true;
}

Stripe *processContour(OutputArrayOfArrays contour) {
	Stripe *s;
	Point2f box[4];

	auto rect = minAreaRect(contour);
	rect.points(box);
	s = new(Stripe);
	if (rect.size.height < rect.size.width) {
		s->width = rect.size.height;
		s->length = rect.size.width;
		s->angle = 90 + rect.angle;
		s->box[0] = box[2];
		s->box[1] = box[3];
		s->box[2] = box[1];
		s->box[3] = box[0];
	} else {
		s->width = rect.size.width;
		s->length = rect.size.height;
		s->angle = rect.angle;
		s->box[0] = box[1];
		s->box[1] = box[2];
		s->box[2] = box[0];
		s->box[3] = box[3];
	}

	return s;
}

/* It looks that this is worse than the minAreaRect, at least at the moment...

Stripe *processContour(OutputArrayOfArrays contour) {
	Stripe *s = NULL;
	double p;
	Point2f box[4];
	vector<Point> approx;

	p = arcLength(contour, true);

	// we don't care about contours that are too small
	if (p < 40)
		return NULL;

	approxPolyDP(contour, approx, p * 0.04, true);
	if (approx.size() != 4)
		return NULL;

	s = new(Stripe);
	for(int i = 0; i < approx.size(); i++) {
		Point pt = approx[i];

		box[i].x = pt.x;
		box[i].y = pt.y;
	}
	box[4] = box[0];

	qsort(box, 4, sizeof(box[0]), cmppoint);
	if (box[0].x < box[1].x) {
		s->box[0] = box[0];
		s->box[1] = box[1];
	} else {
		s->box[0] = box[1];
		s->box[1] = box[0];
	}

	if (box[2].x < box[3].x) {
		s->box[2] = box[2];
		s->box[3] = box[3];
	} else {
		s->box[2] = box[3];
		s->box[3] = box[2];
	}

	// calculate the length and the width of the stripe
	s->width = (distance(s->box[0], s->box[1]) + distance(s->box[2], s->box[3])) / 2.0;
	s->length = (distance(s->box[1], s->box[2]) + distance(s->box[3], s->box[0])) / 2.0;

	// calculate the angles of the two sides of the rectangle
	double a1 = (atan2((s->box[2].x - s->box[0].x), (s->box[2].y - s->box[0].y))/M_PI) * 180.;
	double a2 = (atan2((s->box[3].x - s->box[1].x), (s->box[3].y - s->box[1].y))/M_PI) * 180.;

	// and the average of the angles
	s->angle = (a1 + a2) / 2.0;

	return s;
}
*/

int stripecmp(const void *p1, const void *p2) {
	int ret;

	Stripe **sp1 = (Stripe **) p1;
	Stripe **sp2 = (Stripe **) p2;

	Stripe *s1 = *sp1;
	Stripe *s2 = *sp2;

	return s1->box[0].x - s2->box[0].x;
}

bool checkTarget(Stripe* s1, Stripe* s2) {
	double a1, a2;

	a1 = s1->area();
	a2 = s2->area();

	// Check the angles
//	printf("\tangles %f:%f %f\n", si->angle, sj->angle, si->angle - sj->angle);
	if (s1->angle*s2->angle > 0)
		return false;

	// If the areas are significantly different, skip it
	double ar = a1 / a2;
	if (ar < 0.25 || ar > 4.0)
		return false;

	// If the tops of the stripes are too far apart, probably not part of the same target
	// Check this only if the stripes aer not too small
	if (s1->length > 5 && abs(s1->box[0].y - s2->box[0].y) > s1->length / 4)
		return false;

	// If the bottom of 2 is too close to the top of 1, skip it
	if (s1->box[0].y > s2->box[3].y)
		return false;


	// If the top of 2 is too close to the bottom of 1, skip it
	if (s1->box[3].y < s2->box[0].y)
		return false;

//	printf("\tdistance %f length%f\n", sj->box[0].x - si->box[0].x, si->length);
	// If the x-distance is much further than the length, skip it
	// This doesn't make much sense, because it looks that the stripes from different targets
	// are actually closer. So I am commenting it out.
	if ((s2->box[0].x - s1->box[0].x) / s1->length > 5)
		return false;


	
	return true;
}

void processTargets(Mat &src, Mat &dst, const CameraConfig& c) {
	Mat mask, hsv, gauss, emask, dmask;
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours;
	int stripenum;
	Stripe *stripes[64];

/*
	// Gauss filter for smoothing the imag
	GaussianBlur(src, gauss, Size(11, 11), 1.5);
*/

	// Convert to HSV
	cvtColor(src /*gauss*/, hsv, CV_BGR2HSV);

	// Get only pixels that have the colors we expect the stripes to be
	inRange(hsv, Scalar(c.targetHLow, c.targetSLow, c.targetVLow), Scalar(c.targetHHigh, c.targetSHigh, c.targetVHigh), mask);

	// Remove small blobs
//	erode(mask, emask, erosionElement, Point(-1, -1), 2);
//	dilate(emask, dmask, erosionElement, Point(-1, -1), 2);

	// Find contours
	findContours(mask/*dmask*/, contours, hierarchy, RETR_TREE, CHAIN_APPROX_TC89_KCOS/*CHAIN_APPROX_SIMPLE*/);

	// Identify stripes
	stripenum = 0;
	for(size_t i = 0; i < contours.size(); i++) {
		if (hierarchy[i][3] >= 0)
			continue;

		Stripe *st = processContour(contours[i]);
		if (st == NULL) {
//			printf("\tbad\n");
			continue;
		}

		// check if the stripe follows some common sense restrictions
		if (!checkStripe(st)) {
//			st->draw(dst, 2);
			delete(st);
			continue;
		}

		stripes[stripenum++] = st;
		if (stripenum >= 64) {
			wpi::outs() << "*** stripes > 64\n";
			break;
		}
	}

//	printf("stripenum %d\n", stripenum);
	if (stripenum < 2) {
		return;
	}

	// sort horizontally
	qsort(stripes, stripenum, sizeof(Stripe *), stripecmp);

	// try to combine stripes into vision targets
	vector<Target *> targets;
	for(int i = 0; i < stripenum - 1; i++) {
		Stripe *si = stripes[i];

		if (si == NULL)
			continue;

//		printf("stripe %d angle %f\n", i, si->angle);
		for(int j = i + 1; j < stripenum; j++) {
			Stripe *sj = stripes[j];
			if (sj == NULL)
				continue;

			if (!checkTarget(si, sj))
				continue;

			targets.push_back(new Target(si, sj));
			stripes[i] = NULL;
			stripes[j] = NULL;
			break;
		}

		if (stripes[i]) {
			delete(stripes[i]);
			stripes[i] = NULL;
		}
	}

	// find the target closest to the center
	Target *center = NULL;
	int w2 = c.width / 2;
	int bestdx = w2;
	for(size_t i = 0; i < targets.size(); i++) {
		Target *t = targets[i];
		int dx = abs(t->left->box[0].x - w2);

		if (dx < bestdx) {
			center = t;
			bestdx = dx;
		}
	}

	for(size_t i = 0; i < targets.size(); i++) {
		char buf[32];
		Target *t;

		t = targets[i];
		t->left->draw(dst, 0, t==center?3:1);
		t->right->draw(dst, 1, t==center?3:1);

		if (t == center) {
			target_table->PutNumber("X", t->tpos[0]);
			target_table->PutNumber("Y", t->tpos[1]);
			target_table->PutNumber("Z", t->tpos[2]);
			target_table->PutNumber("Yaw", t->yaw);
			target_table->PutNumber("Pitch", t->pitch);
			target_table->PutNumber("Roll", t->roll);
		}
						
/*
		snprintf(buf, sizeof(buf), "%02d", i);
		std::shared_ptr<nt::NetworkTable> ttbl = target_table->GetSubTable(buf);
		ttbl->PutNumber("X", t->tpos[0]);
		ttbl->PutNumber("Y", t->tpos[1]);
		ttbl->PutNumber("Z", t->tpos[2]);
		ttbl->PutNumber("Yaw", t->yaw);
		ttbl->PutNumber("Pitch", t->pitch);
		ttbl->PutNumber("Roll", t->roll);
*/

#if DEBUG
		printf("Target %d\n", i);
		printf("\tyaw %f pitch %f roll %f\n", t->yaw, t->pitch, t->roll);
		printf("\tx %f y %f z %f\n", t->tpos[0], t->tpos[1], t->tpos[2]);
#endif
	}

/*
	// (hopefully) clear all stale sub-tables
	for(size_t i = targets.size(); i < 10; i++) {
		char buf[32];

		snprintf(buf, sizeof(buf), "%02d", i);
		target_table->Delete(buf);
	}
*/

	// Cleanup
	for(size_t i = 0; i < targets.size(); i++)
		delete(targets[i]);
}

void processLine(Mat &src, Mat &dst, const CameraConfig& c) {
	Mat gray, mask, rsz;
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours;
	Stripe *best;

	// convert the image to grayscale
	cvtColor(src, gray, CV_BGR2GRAY);

	// make the image black/white with a dynamic Otsu thresholding
	threshold(gray, mask, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	// find all contours in the image (hopefully only one)
	findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_TC89_KCOS/*CHAIN_APPROX_SIMPLE*/);

	// convert the contours into stripes and find the biggest one by area (hopefully the white line)
	int bestidx = -1;
	double bestarea = 0.0;
	for(size_t i = 0; i < contours.size(); i++) {
		double area = contourArea(contours[i]);
		if (area > bestarea) {
			bestidx = i;
			bestarea = area;
		}
	}

	if (bestidx < 0)
		return;

	best = processContour(contours[bestidx]);
/*
	best = NULL;
	for(size_t i = 0; i < contours.size(); i++) {
		if (contours[i].area() > 0)
			continue;

		Stripe *st = processContour(contours[i]);
		if (st == NULL)
			continue;

		if (best == NULL || best->area() < st->area()) {
			delete best;
			best = st;
		} else {
			delete st;
		}
	}

	// if we didn't find any, just return
	if (!best)
		return;
*/

	// draw the ractangle on the image
	best->draw(dst, 0, 1);

	// put the angle and displacement in the NetworkTable
	line_table->PutNumber("Angle", best->angle);

	double center = c.lineCoeff * (((best->box[0].x + best->box[1].x + best->box[2].x + best->box[3].x) / 4) / c.width - 0.5);
	line_table->PutNumber("X", center);

#if DEBUG
	printf("angle %f x %f\n", best->angle, center);
#endif
	delete best;
}

wpi::raw_ostream& ParseError() {
	return wpi::errs() << "config error in '" << configFile << "': ";
}

bool ReadCameraConfig(const wpi::json& config) {
	CameraConfig c;

	c.config = config;
	c.width = 640;
	c.height = 480;
	c.targetTrack = false;
	c.targetHLow = 0;
	c.targetSLow = 0;
	c.targetVLow = 0;
	c.targetHHigh = 180;
	c.targetSHigh = 255;
	c.targetVHigh = 255;
	c.lineTrack = false;
	c.lineCoeff = 1.0;

	// name
	try {
		c.name = config.at("name").get<string>();
	} catch (const wpi::json::exception& e) {
		ParseError() << "could not read camera name: " << e.what() << '\n';
		return false;
	}

	// path
	try {
		c.path = config.at("path").get<string>();
	} catch (const wpi::json::exception& e) {
		ParseError() << "camera '" << c.name << "': could not read path: " << e.what() << '\n';
		return false;
	}

	// image width and height
	try {
		c.width = config.at("width").get<int>();
		c.height = config.at("height").get<int>();
	} catch (const wpi::json::exception& e) {
		// ignore, we have default values
	}

	try {
		for (auto&& prop : config.at("properties")) {
			std::string name;

			name = prop.at("name").get<std::string>();
			if (name == "track_target") {
				c.targetTrack  = prop.at("value").get<bool>();
			} else if (name == "track_line") {
				c.lineTrack = prop.at("value").get<bool>();
			} else if (name == "h_low") {
				c.targetHLow = prop.at("value").get<int>();
			} else if (name == "s_low") {
				c.targetSLow = prop.at("value").get<int>();
			} else if (name == "v_low") {
				c.targetVLow = prop.at("value").get<int>();
			} else if (name == "h_high") {
				c.targetHHigh = prop.at("value").get<int>();
			} else if (name == "s_high") {
				c.targetSHigh = prop.at("value").get<int>();
			} else if (name == "v_high") {
				c.targetVHigh = prop.at("value").get<int>();
			} else if (name == "line_coeff") {
				c.lineCoeff = prop.at("value").get<double>();
			}
		}
	} catch (const wpi::json::exception& e) {
		ParseError() << "could not read property name: " << e.what();
		// ignore
	}

	cameras.emplace_back(move(c));
	return true;
}

bool ReadConfig() {
	// open config file
	error_code ec;
	wpi::raw_fd_istream is(configFile, ec);
	if (ec) {
		wpi::errs() << "could not open '" << configFile << "': " << ec.message() << '\n';
		return false;
	}

	// parse file
	wpi::json j;
	try {
		j = wpi::json::parse(is);
	} catch (const wpi::json::parse_error& e) {
		ParseError() << "byte " << e.byte << ": " << e.what() << '\n';
		return false;
	}

	// top level must be an object
	if (!j.is_object()) {
		ParseError() << "must be JSON object\n";
		return false;
	}

	// team number
	try {
		team = j.at("team").get<unsigned int>();
	} catch (const wpi::json::exception& e) {
		ParseError() << "could not read team number: " << e.what() << '\n';
		return false;
	}

	// ntmode (optional)
	if (j.count("ntmode") != 0) {
		try {
			auto str = j.at("ntmode").get<string>();
			wpi::StringRef s(str);
			if (s.equals_lower("client")) {
				server = false;
			} else if (s.equals_lower("server")) {
				server = true;
			} else {
				ParseError() << "could not understand ntmode value '" << str << "'\n";
			}
		} catch (const wpi::json::exception& e) {
			ParseError() << "could not read ntmode: " << e.what() << '\n';
		}
	}

	// cameras
	try {
		for (auto&& camera : j.at("cameras")) {
			if (!ReadCameraConfig(camera)) return false;
		}
	} catch (const wpi::json::exception& e) {
		ParseError() << "could not read cameras: " << e.what() << '\n';
		return false;
	}

	return true;
}

void CameraThread(CameraConfig* config) {
	CS_Status status = 0;
	char buf[20];
	cv::Mat mat;

	wpi::outs() << "Starting camera '" << config->name << "' on " << config->path << '\n';
	auto camera = frc::CameraServer::GetInstance()->StartAutomaticCapture(config->name, config->path);
	camera.SetConfigJson(config->config);

	if (!config->targetTrack && !config->lineTrack)
		return;

	string tgtname;
	if (config->targetTrack)
		tgtname = config->name + ":Target";
	else
		tgtname = config->name + ":Line";

	cs::CvSink cvSink = frc::CameraServer::GetInstance()->GetVideo(camera);
	cs::CvSource outputStream = frc::CameraServer::GetInstance()->PutVideo(tgtname, config->width, config->height);
	outputStream.CreateProperty("track_target", cs::VideoProperty::Kind::kBoolean, 0, 1, 1, 0, config->targetTrack);
	outputStream.CreateProperty("track_line", cs::VideoProperty::Kind::kBoolean, 0, 1, 1, 0, config->lineTrack);
	outputStream.CreateProperty("h_low", cs::VideoProperty::Kind::kInteger, 0, 180, 1, 25, config->targetHLow);
	outputStream.CreateProperty("s_low", cs::VideoProperty::Kind::kInteger, 0, 255, 1, 128,config->targetSLow);
	outputStream.CreateProperty("v_low", cs::VideoProperty::Kind::kInteger, 0, 255, 1, 128, config->targetVLow);

	outputStream.CreateProperty("h_high", cs::VideoProperty::Kind::kInteger, 0, 180, 1, 35, config->targetHHigh);
	outputStream.CreateProperty("s_high", cs::VideoProperty::Kind::kInteger, 0, 255, 1, 128,config->targetSHigh);
	outputStream.CreateProperty("v_high", cs::VideoProperty::Kind::kInteger, 0, 255, 1, 128, config->targetVHigh);

//	int lcoeff = outputStream.CreateProperty("line_coeff", cs::VideoProperty::Kind::kString, 0, 255, 1, 128, 0);
	snprintf(buf, sizeof(buf), "%f", config->lineCoeff);
	outputStream.CreateStringProperty("line_coeff", buf);
	
	// Adding them here so it is easier to copy/paste the settings
//	outputStream.CreateProperty("exposure_auto", cs::VideoProperty::Kind::kInteger, 0, 1, 1, 1, 1);
//	outputStream.CreateProperty("exposure_absolute", cs::VideoProperty::Kind::kInteger, 0, 1, 1, 0, 0);

	cs::AddListener(
		[&](const cs::RawEvent& event) {
#if DEBUG
			cout << "Property updated '" << event.name << "' value: " << event.value << "\n";
#endif
			if (event.name == "h_low") {
				config->targetHLow = event.value;
			} else if (event.name == "s_low") {
				config->targetSLow = event.value;
			} else if (event.name == "v_low") {
				config->targetVLow = event.value;
			} else if (event.name == "h_high") {
				config->targetHHigh = event.value;
			} else if (event.name == "s_high") {
				config->targetSHigh = event.value;
			} else if (event.name == "v_high") {
				config->targetVHigh = event.value;
			} else if (event.name == "track_target") {
				config->targetTrack = event.value;
			} else if (event.name == "track_line") {
				config->lineTrack = event.value;
			} else if (event.name == "line_coeff") {
				char *s;
				double v;

				v = strtod(event.valueStr.c_str(), &s);
				if (*s == '\0') {
					config->lineCoeff = v;
				}
			} else if (event.name == "exposure_auto") {
				// ignore for now
			} else if (event.name == "exposure_absolute") {
				// ignore for now
				// camera.SetExposureManual(event.value);
			}
		}, cs::RawEvent::kSourcePropertyValueUpdated, true, &status);

	uint64_t prevtime = 0;
	int framecount = 0;
	while (true) {
		// Tell the CvSink to grab a frame from the camera and put it
		// in the source mat.  If there is an error notify the output.
		uint64_t tstamp = cvSink.GrabFrame(mat);
		if (tstamp == 0) {
			// Send the output the error.
			outputStream.NotifyError(cvSink.GetError());
			// skip the rest of the current iteration
			continue;
		}

		if (config->targetTrack)
			processTargets(mat, mat, *config);

		if (config->lineTrack)
			processLine(mat, mat, *config);

		if ((tstamp - prevtime) > 1000000) {
#if DEBUG
			if (prevtime != 0)
				printf("Rate: %f fps\n", framecount*1000000.0 / (tstamp - prevtime));
#endif
			framecount = 0;
			prevtime = tstamp;
		}

		// Give the output stream a new image to display
		outputStream.PutFrame(mat);
		framecount++;
	}
}

int main(int argc, char* argv[]) {
	if (argc >= 2)
		configFile = argv[1];

	erosionElement = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));

	// read configuration
	if (!ReadConfig())
		return EXIT_FAILURE;

	// start NetworkTables
	ntinst = nt::NetworkTableInstance::GetDefault();
	if (server) {
		wpi::outs() << "Setting up NetworkTables server\n";
		ntinst.StartServer();
	} else {
		wpi::outs() << "Setting up NetworkTables client for team " << team << '\n';
		ntinst.StartClientTeam(team);
	}

	target_table = ntinst.GetTable("Target");
	line_table = ntinst.GetTable("Line");

	// start cameras
	for(int i = 0; i < cameras.size(); i++) {
		auto camera = &cameras[i];
//	for (auto&& camera : cameras) {
		thread cameraThread(CameraThread, camera);
		cameraThread.detach();
	}

	// loop forever
	for (;;)
		this_thread::sleep_for(chrono::seconds(10));
}
