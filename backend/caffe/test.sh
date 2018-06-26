#!/bin/sh
../caffe_o/build/tools/caffe test \
	-model  ./models/DeResnet20/trainval.prototxt \
	-weights ./snapshot/DeResnet20-NEW_iter_5000.caffemodel \
	$@
