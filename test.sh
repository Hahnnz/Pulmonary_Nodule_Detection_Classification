<<<<<<< HEAD
set -e

./caffe_d/build/tools/caffe test \
	-model ./models/Deconv-model2/p1-train1.prototxt \
	-weights ./snapshot/Deconv-model2_iter_4000.caffemodel \
=======
#!/bin/sh
set -e

./caffe_o/build/tools/caffe test \
	-model  ./models/CNN10/analysis.prototxt \
	-weights ./snapshot/Cls-state/CNN-model10_iter_7100.caffemodel \
	-gpu 0
>>>>>>> a97955964cb8e3a5556bffe246638df0f3a6d22c
	$@
