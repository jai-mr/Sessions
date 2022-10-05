#reference : https://dev.to/flpslv/using-makefiles-to-build-and-publish-docker-containers-7c8

#Dockerfile vars
pythonver=3.8-slim

#vars
IMAGENAME=emlo2_s4
#REPO= jaideepmr
APP_VER = 1.0
#IMAGEFULLNAME=${REPO}/${IMAGENAME}:${APP_VER}

IMAGEFULLNAME=${IMAGENAME}:${APP_VER}

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
#	    @docker build --pull --build-arg PY_VER=${pythonver} -t ${IMAGEFULLNAME} .  --shm-size=256m
	    @docker build --build-arg PY_VER=${pythonver} -t ${IMAGEFULLNAME} . 

push:
	    @docker push ${IMAGEFULLNAME}

all: build push

train: ## Train the model
	python src/train.py

debug: ## Enter debugging mode with pdb
	#
	# tips:
	# - use "import pdb; pdb.set_trace()" to set breakpoint
	# - use "h" to print all commands
	# - use "n" to execute the next line
	# - use "c" to run until the breakpoint is hit
	# - use "l" to print src code around current line, "ll" for full function code
	# - docs: https://docs.python.org/3/library/pdb.html
	#
	python -m pdb src/train.py debug=default

# Makefile for docker build

