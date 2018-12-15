#!/bin/sh
GLOG_logtostderr=1 ../../build/tools/caffe train \
  --solver=solver.prototxt \
  --weights  bvlcalexnet.caffemodel\
  --gpu=0 2>&1 | tee log.txt
echo 'Done.'
