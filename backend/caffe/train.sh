set +x

NET=DeResnet20

mkdir ./logs/${NET}

LOG="./logs/${NET}/${NET}_`date +'%Y-%m-%d_%H-%M-%S'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

../caffe_o/build/tools/caffe train \
	-gpu=0 \
	-sighup_effect stop \
	-solver=./models/$NET/solver.prototxt
