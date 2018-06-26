#!/usr/bin/env sh

<<<<<<< HEAD
FILE_NAME=make_image_mean.sh
WORK_DIR=.
CAFFE_ROOT_DIR=./caffe_o
TOOLS=$CAFFE_ROOT_DIR/build/tools
DBTYPE=lmdb

$TOOLS/compute_image_mean -backend=$DBTYPE \
	$WORK_DIR/lmdb/train_lmdb $WORK_DIR/train_mean.binaryproto

=======
./caffe_o/build/tools/compute_image_mean -backend=lmdb \
	./lmdb/train_Cls_lmdb ./train_mean.binaryproto

./caffe_o/build/tools/compute_image_mean -backend=lmdb \
	./lmdb/val_Cls_lmdb ./val_mean.binaryproto
>>>>>>> a97955964cb8e3a5556bffe246638df0f3a6d22c

#$TOOLS/compute_image_mean -backend=$DBTYPE \
#	$WORK_DIR/p1_val_lmdb $WORK_DIR/p1_val_mean.binaryproto

echo "연산 완료."
