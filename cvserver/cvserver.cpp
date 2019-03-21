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
#include <sys/stat.h>
#include <unistd.h>

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

#include "csrv.h"

int main(int argc, char* argv[]) {
	if (csrvInit(NULL, NULL) != 0)
		return -1;

	// loop forever
	for (;;)
		sleep(10);
}
