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
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d/calib3d_c.h>
#include <networktables/NetworkTableInstance.h>
#include <wpi/StringRef.h>
#include <wpi/json.h>
#include <wpi/raw_istream.h>
#include <wpi/raw_ostream.h>

#include "cameraserver/CameraServer.h"

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
using namespace cv::xfeatures2d;

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
	Scalar		targetColorLow;
	Scalar		targetColorHigh;

	// line tracking
	bool		lineTrack;
	double		lineCoeff;	// used to calculate real distance

	wpi::json	config;
};

vector<CameraConfig> cameras;

nt::NetworkTableInstance ntinst;
std::shared_ptr<nt::NetworkTable> table;

struct Stripe {
	Point2f	box[4];
	double	length;
	double	width;
	double	angle;

//	bool left() { return angle < 0.0; }
	void draw(Mat &dst, bool left) {
		Scalar color = left?Scalar(0, 0, 255):Scalar(255, 0, 0);

		line(dst, box[0], box[1], color);
		line(dst, box[1], box[3], color);
		line(dst, box[3], box[2], color);
		line(dst, box[2], box[0], color);
	}

	double area() {
		return length * width;
	}
};

struct Target {
	Stripe* left;
	Stripe* right;
	Mat	h;
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
const double angle_low = -18.0;
const double angle_high = 18.0;

// Model of the target. All numbers are in centimeters
vector<Point3f> Target::model = {
	// left stripe
	Point3f(-15.3967/7, 99.3775/7, 0),
	Point3f(-10.16/7, 98.1442/7, 0),
	Point3f(-18.8945/7, 85.8525/7, 0),
	Point3f(-13.6578/7, 84.6192/7, 0),

	// right stripe
	Point3f(15.3967/7, 99.3775/7, 0),
	Point3f(10.16/7, 98.1442/7, 0),
	Point3f(18.8945/7, 85.8525/7, 0),
	Point3f(13.6578/7, 84.6192/7, 0),
};

// Raspberry Pi Camera (640x480)
Mat Target::cameraMatrix = (Mat_<double>(3,3) <<  4.6676194913596777e+02, 0., 3.1086529034645207e+02, 0., 4.6676194913596777e+02, 2.3946292934201807e+02, 0., 0., 1.);

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

	Point2f *pt1 = (Point2f *) p1;
	Point2f *pt2 = (Point2f *) p2;

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

	// check if the length and the width are within expected ratio range
	if (isnan(r) || r < (stripe_ratio - stripe_err) || r > (stripe_ratio + stripe_err))
		return false;

	// check if the angle of the stripe is within the expected range
	if (s->angle < angle_low || s->angle > angle_high)
		return false;

	// check if the stripe is big enough
	if (s->area() < 50)
		return false;

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
	if (s1->angle - s2->angle < 0)
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
	if ((s2->box[0].x - s1->box[0].x) / s1->length > 10)
		return false;


	return true;
}

void processTargets(Mat &src, Mat &dst, const CameraConfig& c) {
	Mat mask, hsv;
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours;
	int stripenum;
	Stripe *stripes[64];

//	cvtColor(src, hsv, CV_BGR2HSV);

	// Get only pixels that have the colors we expect the stripes to be
	inRange(src, c.targetColorLow, c.targetColorHigh, mask);

	// Find contours
	findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_TC89_KCOS/*CHAIN_APPROX_SIMPLE*/);

	// Identify stripes
	stripenum = 0;
	for(size_t i = 0; i < contours.size(); i++) {
		Stripe *st = processContour(contours[i]);

		// check if the stripe follows some common sense restrictions
		if (!checkStripe(st)) {
			delete(st);
			continue;
		}

//		s.draw(dst, true);
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

	for(size_t i = 0; i < targets.size(); i++) {
		char buf[32];
		Target *t;

		t = targets[i];
		t->left->draw(dst, true);
		t->right->draw(dst, false);

		snprintf(buf, sizeof(buf), "%02d", i);
		std::shared_ptr<nt::NetworkTable> ttbl = table->GetSubTable(buf);
		ttbl->PutNumber("X", t->tpos[0]);
		ttbl->PutNumber("Y", t->tpos[1]);
		ttbl->PutNumber("Z", t->tpos[2]);
		ttbl->PutNumber("Yaw", t->yaw);
		ttbl->PutNumber("Pitch", t->pitch);
		ttbl->PutNumber("Roll", t->roll);

		printf("Target %d\n", i);
		printf("\tyaw %f pitch %f roll %f\n", t->yaw, t->pitch, t->roll);
		printf("\tx %f y %f z %f\n", t->tpos[0], t->tpos[1], t->tpos[2]);
		printf("\t%f %f\n", t->left->angle, t->right->angle);
	}

	// (hopefully) clear all stale sub-tables
	for(size_t i = targets.size(); i < 10; i++) {
		char buf[32];

		snprintf(buf, sizeof(buf), "%02d", i);
		table->Delete(buf);
	}

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
	best = NULL;
	for(size_t i = 0; i < contours.size(); i++) {
		Stripe *st = processContour(contours[i]);
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

	// draw the ractangle on the image
	best->draw(dst, true);

	// put the angle and displacement in the NetworkTable
	table->PutNumber("Angle", best->angle);

	double center = c.lineCoeff * (((best->box[0].x + best->box[1].x + best->box[2].x + best->box[3].x) / 4) / c.width - 0.5);
	table->PutNumber("X", center);

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
	c.targetColorLow = Scalar(0, 245, 0);
	c.targetColorHigh = Scalar(255, 255, 255);
	c.lineTrack = false;
	c.lineCoeff = 1.0;
//	c.targetColorLow = Scalar(60-10, 0, 30);
//	c.targetColorHigh = Scalar(60+10, 255, 255);

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
			} else if (name == "target_rgb_low" || name == "target_rgb_high") {
				int clr[3];

				int i = 0;
				cout << "property " << name << "\n";
				for(auto && p : prop.at("value")) {
					clr[i] = p.get<int>();
					i++;
					if (i > 3)
						break;
				}

				Scalar sclr(clr[0], clr[1], clr[2]);
				if (name == "target_rgb_low")
					c.targetColorLow = sclr;
				else
					c.targetColorHigh = sclr;
			} else if (name == "track_line") {
				c.lineTrack = prop.at("value").get<bool>();
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

void CameraThread(const CameraConfig& config) {
	cv::Mat mat;

	wpi::outs() << "Starting camera '" << config.name << "' on " << config.path << '\n';
	auto camera = frc::CameraServer::GetInstance()->StartAutomaticCapture(config.name, config.path);
	camera.SetConfigJson(config.config);

	if (!config.targetTrack && !config.lineTrack)
		return;

	cs::CvSink cvSink = frc::CameraServer::GetInstance()->GetVideo();
	cs::CvSource outputStream = frc::CameraServer::GetInstance()->PutVideo("Target", config.width, config.height);

	while (true) {
		// Tell the CvSink to grab a frame from the camera and put it
		// in the source mat.  If there is an error notify the output.
		if (cvSink.GrabFrame(mat) == 0) {
			// Send the output the error.
			outputStream.NotifyError(cvSink.GetError());
			// skip the rest of the current iteration
			continue;
		}

		if (config.targetTrack)
			processTargets(mat, mat, config);

		if (config.lineTrack)
			processLine(mat, mat, config);

		// Give the output stream a new image to display
		outputStream.PutFrame(mat);
	}
}

int main(int argc, char* argv[]) {
	if (argc >= 2)
		configFile = argv[1];

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

	table = ntinst.GetTable("Target");

	// start cameras
	for (auto&& camera : cameras) {
		thread cameraThread(CameraThread, camera);
		cameraThread.detach();
	}

	// loop forever
	for (;;)
		this_thread::sleep_for(chrono::seconds(10));
}
