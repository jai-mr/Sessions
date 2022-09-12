#reference : https://dev.to/flpslv/using-makefiles-to-build-and-publish-docker-containers-7c8

#Dockerfile vars
pythonver=3.8-slim

#vars
IMAGENAME=emlo2_s2
REPO= jaideepmr
APP_VER = 1.0
IMAGEFULLNAME=${REPO}/${IMAGENAME}:${APP_VER}

.PHONY: help build push all

help:
	    @echo "Makefile arguments:"
	    @echo ""
	    @echo "Makefile commands:"
	    @echo "build"
	    @echo "push"
	    @echo "all"

.DEFAULT_GOAL := all

build:
	    @docker build --build-arg PY_VER=${pythonver} -t ${IMAGEFULLNAME} . --shm-size=256m

push:
	    @docker push ${IMAGEFULLNAME}

all: build push
