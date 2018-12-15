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
void diskmeans_pre(Dtype* dis_vec,  Dtype* k_vec, float* kernel2, int num, int count){
  float kernel_new[num];
  int kernel_num[num];
  // Clustering
  for (int i=0; i<count; i++){
    int tmp = 0;
    float score = 0;

    score=(dis_vec[i]-kernel2[0])*(dis_vec[i]-kernel2[0]);

    for(int j=1; j<num; j++){
      if((dis_vec[i]-kernel2[j])*(dis_vec[i]-kernel2[j])<score){
        tmp=j;score=(dis_vec[i]-kernel2[j])*(dis_vec[i]-kernel2[j]);
      }
    }
  
    k_vec[i] = tmp;
  }
  // Update kernel
  for(int i=0;i<num;i++){kernel_num[i]=0;kernel_new[i]=0;}
  for(int i=0; i<count; i++){
    kernel_num[int(k_vec[i])] +=1;
    kernel_new[int(k_vec[i])] +=dis_vec[i];
  }
  for(int i=0; i<num; i++){
  if(kernel_num[i]>0) kernel2[i] = kernel_new[i] / kernel_num[i];}
}
template <typename Dtype>
void disKmeans(Dtype* data_vec, Dtype* T_vec, Dtype* k_vec, Dtype* dis_vec, float* kernel1, float* kernel2, int num, int count){
  int epoch = 10;
  float n_max=0;
  float n_min=220000;
  int kernel_num[num];
 for(int i=0; i<count; i++){
    if (dis_vec[i]>0 && dis_vec[i]>n_max) n_max = dis_vec[i];
    else if(dis_vec[i]<0 && -1*dis_vec[i]>n_max) n_max = -1*dis_vec[i];
    //if(fabs(data_vec[i])>n_max) n_max=data_vec[i];
  }
  for(int i=0; i<count; i++){
    if (dis_vec[i]>0 && dis_vec[i]<n_min) n_min = dis_vec[i];
    else if(dis_vec[i]<0 && -1*dis_vec[i]<n_min) n_min = -1*dis_vec[i];
    //if(fabs(data_vec[i])>n_max) n_max=data_vec[i];
  }
  n_max=pow(2,floor(log(n_max)/log(2)));
  
  // Set kernel
  for(int i=0; i<num; i=i+1){
    //if(n_max==128) 
    kernel2[i]=n_max-((n_max-n_min)/(num-1))*i;
    //else
    //kernel2[i] = n_max/pow(2,i);
    //kernel[i] = n_max - floor(i/2)*n_max/(bit_num-1);
    //kernel[i+1] = -kernel[i];
  }

  for(int e=0; e<epoch; e++){
    diskmeans_pre(dis_vec, k_vec, kernel2, num, count);
  }
  for(int i=0;i<num;i++) kernel_num[i]=0;
  for(int i=0; i<count; i++){
    kernel_num[int(k_vec[i])] +=1;
  }

  /*for(int i=0; i<count; i++){
    if(dis_vec[i] == 0){
      T_vec[i] = 0;
      mmm+=1;
    }
  }*/
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
