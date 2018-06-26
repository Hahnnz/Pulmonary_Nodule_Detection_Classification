set -e

./caffe_o/build/tools/caffe train \
	--solver=./models/CNN5/solver.prototxt \
	--snapshot=./snapshot/Cls-state/CNN-model4_iter_50000.caffemodel
