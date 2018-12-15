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
void kmeans_pre(float* data_vec,  int* data_index, float* kernel, int count);

void Kmeans(float* data_vec, int* data_index, float* kernel, int bit_num, int count){
  int num = 9;
  int epoch = 10;
  float n_max=0;
  std::ofstream fout("/home/min/tmp.txt");
  fout<<n_max<<std::endl;
  for(int i=0; i<count; i++){
    if(fabs(data_vec[i])>n_max) n_max=data_vec[i];
  }
  n_max=pow(2,floor(log(n_max)/log(2)));
  fout<<fabs(n_max)<<std::endl;
  fout<<data_vec[0]<<std::endl;
  
  // Set kernel
  kernel[0] = 0.0;
  for(int i=1; i<2*bit_num-1; i=i+2){
    kernel[i] = n_max/pow(2,i-1);
    //kernel[i] = n_max - floor(i/2)*n_max/(bit_num-1);
    kernel[i+1] = -kernel[i];
  }

  srand(time(0));
  //kernel[9] = 0.1;
  //for(int i=2*bit_num-1; i<num; i++) kernel[i] = 2.0*n_max*rand()/INT_MAX-n_max;
  // Kmeans
  for(int e=0; e<epoch; e++){
    kmeans_pre(data_vec, data_index, kernel, count);
  }

}

void kmeans_pre(float* data_vec, int* data_index, float* kernel, int count){
  int num = 9;
  float kernel_new[num];
  int kernel_num[num];
  // Clustering
  for (int i=0; i<count; i++){
    int tmp = 0;
    float score = fabs(data_vec[i]-kernel[tmp]);
    for(int j=1; j<num; j++)
      if(fabs(data_vec[i]-kernel[j])<score){tmp=j;score=fabs(data_vec[i]-kernel[j]);}
    data_index[i] = tmp;
  }
  // Update kernel
  for(int i=9;i<num;i++){kernel_num[i]=0;kernel_new[i]=0;}
  for(int i=0; i<count; i++){
    kernel_num[data_index[i]] +=1;
    kernel_new[data_index[i]] +=data_vec[i];
  }
  for(int i=3; i<num; i++) kernel[i] = kernel_new[i] / kernel_num[i];
}
}
