#!/usr/bin/env sh
set -e

FILE_NAME=convert_lmdb.sh
WORK_DIR=.
CAFFE_ROOT_DIR=./caffe_o
DATA=$WORK_DIR/dataset
TOOLS=$CAFFE_ROOT_DIR/build/tools

RESIZE=false
if $RESIZE; then
	RESIZE_HEIGHT=64
	RESIZE_WIDTH=64
else
	RESIZE_HEIGHT=0
	RESIZE_WIDTH=0
fi

echo "Converting train dataset to LMDB"

GLOG_logtostderr=1 $TOOLS/convert_imageset \
<<<<<<< HEAD
	$WORK_DIR \
	$DATA/train_Cls/lmdb.txt \
	$WORK_DIR/lmdb/train_Cls_lmdb
=======
	--shuffle \
	$WORK_DIR \
	$DATA/train_Cls/LMDB.txt \
	$WORK_DIR/lmdb/train_lmdb
>>>>>>> a97955964cb8e3a5556bffe246638df0f3a6d22c

echo "Converting validation dataset to LMDB"

 
GLOG_logtostderr=1 $TOOLS/convert_imageset \
<<<<<<< HEAD
	$WORK_DIR \
	$DATA/val_Cls/lmdb.txt \
	$WORK_DIR/lmdb/val_Cls_lmdb
=======
	--shuffle \
	$WORK_DIR \
	$DATA/val_Cls/LMDB.txt \
	$WORK_DIR/lmdb/val_lmdb
>>>>>>> a97955964cb8e3a5556bffe246638df0f3a6d22c

echo "DONE."
