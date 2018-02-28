#!/usr/bin/env sh
set -e

./caffe_d/build/tools/caffe train \
	--solver=./models/Deconv-model2/p1-solver.prototxt $@
/home/dl1/Project/caffe_d/build/tools/caffe train \
	--solver=/home/dl1/Project/models/Deconv-model1/p1-solver.prototxt $@
