#include <algorithm>
#include <string>
#include <vector>
#include <climits>
#include <time.h>
#include <iostream>
#include <fstream>
#include <math.h>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

namespace caffe {

template <typename Dtype>
void disKmeans(Dtype* data_vec, Dtype* T_vec, Dtype* k_vec, Dtype* dis_vec, float* kernel1, float* kernel2, int num, int count){

  Dtype* data_copy=(Dtype*) mallc(count_*sizeof(Dtype));
  caffe_copy
  for(int i=0; i<count; i++){
    if(k_vec[i]<1){
      data_vec[i] = kernel1[int(T_vec[i])];
      T_vec[i] = 0;
    }
    else T_vec[i] = 1;
  }
  for(int i=0; i<count; i++){
    if(dis_vec[i] < 0.000001){
      T_vec[i] = 0;
    }
  }
  std::ofstream fout;
  fout.open("/home/xuyuhui/k_11.txt",ios::app);
  for(int i=0;i<num;i++) fout<<dis_vec[i]<<std::endl<<kernel_num[i]<<std::endl;
  fout.close();
  
}

} //namespace
