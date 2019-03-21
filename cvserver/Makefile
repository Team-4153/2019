EXE=cvserver
DESTDIR?=$(HOME)

OPENCVDIR?=$(HOME)/opencv
FRCDIR=$(HOME)/frc/wpilib
PFDIR=$(HOME)/work/Pathfinder

FRCINC=$(FRCDIR)/include
PFINC=$(PFDIR)/Pathfinder/src/include

COMMONOBJS=csrv.o

.PHONY: clean build install

build: ${EXE}

install:
	sudo install ${EXE} /usr/local/frc/bin
	sudo rm -rf /usr/local/frc/bin/multiCameraServer
	sudo ln -sf /usr/local/frc/bin/${EXE} /usr/local/frc/bin/multiCameraServer

clean:
	rm -f ${EXE} *.o *~

cvserver: cvserver.o $(COMMONOBJS)
	${CXX} -g -o $@ $^ -L$(OPENCVDIR)/lib -lopencv_calib3d -lopencv_flann -lopencv_imgproc -lopencv_core -lopencv_features2d -lopencv_imgcodecs -L$(FRCDIR)/lib -lcameraserver -lcscore -lntcore -lwpiutil -pthread

.cpp.o:
	${CXX} -g -c -o $@ -I$(FRCINC) -I$(FRCINC)/wpiutil -I$(FRCINC)/ntcore -I$(FRCINC)/cscore -I$(FRCINC)/cameraserver -I$(OPENCVDIR)/include/opencv4 -I$(PFINC) -I/usr/include//eigen3 $<
