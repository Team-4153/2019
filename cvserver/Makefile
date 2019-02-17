EXE=cvserver
DESTDIR?=/home/pi/

OPENCVDIR?=/home/pi/opencv
FRCDIR=/home/pi/frc/wpilib

.PHONY: clean build install

build: ${EXE}

install:
	sudo install ${EXE} /usr/local/frc/bin
	sudo rm -rf /usr/local/frc/bin/multiCameraServer
	sudo ln -sf /usr/local/frc/bin/${EXE} /usr/local/frc/bin/multiCameraServer

clean:
	rm -f ${EXE} *.o *~

${EXE}: main.o
	${CXX} -g -o $@ $< -L$(OPENCVDIR)/lib -lopencv_calib3d -lopencv_flann -lopencv_imgproc -lopencv_core -lopencv_features2d -lopencv_imgcodecs -L$(FRCDIR)/lib -lcameraserver -lcscore -lntcore -lwpiutil -pthread

.cpp.o:
	${CXX} -g -c -o $@ -I$(FRCDIR)/include -I$(FRCDIR)/include/wpiutil -I$(FRCDIR)/include/ntcore -I$(FRCDIR)/include/cscore -I$(FRCDIR)/include/cameraserver -I$(OPENCVDIR)/include/opencv4 $<
