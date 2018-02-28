#!/usr/bin/env sh
set -e

./caffe_o/build/tools/caffe train \
	--solver=./models/CNN11/solver.prototxt$@
