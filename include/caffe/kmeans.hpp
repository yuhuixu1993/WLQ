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
void kmeans_pre(Dtype* data_vec,  Dtype* T_vec, float* kernel,int num, int count){
  float kernel_new[num];
  int kernel_num[num];
  // Clustering
  for (int i=0; i<count; i++){
    int tmp = 0;
    float score = 0;
    /*if((data_vec[i]-kernel[0])>0){score=data_vec[i]-kernel[0];}
    else {score=-1*(data_vec[i]-kernel[0]);}
    //float score = fabs(data_vec[i]-kernel[tmp]);*/
    score=(data_vec[i]-kernel[0])*(data_vec[i]-kernel[0]);
    /*for(int j=1; j<num; j++){
      if((data_vec[i]-kernel[j])>0 && (data_vec[i]-kernel[j])<score){tmp=j;score=data_vec[i]-kernel[j];}
      else if ((data_vec[i]-kernel[j])<=0 && (data_vec[i]-kernel[j])>-1*score){tmp=j;score=-1*(data_vec[i]-kernel[j]);}
    }*/
    for(int j=0; j<num; j++){
      if((data_vec[i]-kernel[j])*(data_vec[i]-kernel[j])<score){
        tmp=j;score=(data_vec[i]-kernel[j])*(data_vec[i]-kernel[j]);
      }
    }
      //if(fabs(data_vec[i]-kernel[j])<score){tmp=j;score=fabs(data_vec[i]-kernel[j]);}
    T_vec[i] = tmp;
  }
  // Update kernel
  for(int i=0;i<num;i++){kernel_num[i]=0;kernel_new[i]=0;}
  for(int i=0; i<count; i++){
    kernel_num[int(T_vec[i])] +=1;
    kernel_new[int(T_vec[i])] +=data_vec[i];
  }
  for(int i=1; i<num; i++){//here1
  if(kernel_num[i]>0) kernel[i] = kernel_new[i] / kernel_num[i];}
}
template <typename Dtype>
void Kmeans(Dtype* data_vec, Dtype* T_vec, Dtype* dis_vec, float* kernel, int num, int count){
  int epoch = 10;
  int kernel_num[num];
  float n_max=0;
  float n_min=220000;
  for(int i=0; i<count; i++){
    if (data_vec[i]>0 && data_vec[i]>n_max) n_max = data_vec[i];
    else if(data_vec[i]<0 && -1*data_vec[i]>n_max) n_max = -1*data_vec[i];
    //if(fabs(data_vec[i])>n_max) n_max=data_vec[i];
  }
  for(int i=0; i<count; i++){
    if (data_vec[i]>0 && data_vec[i]<n_min) n_min = data_vec[i];
    else if(data_vec[i]<0 && -1*data_vec[i]<n_min) n_min = -1*data_vec[i];
    //if(fabs(data_vec[i])>n_max) n_max=data_vec[i];
  }
  n_max=pow(2,floor(log(n_max)/log(2)));
  
  // Set kernel
  kernel[0] = 0.0;
  for(int i=1; i<num; i=i+2){
    //if(n_max==128) 
       //kernel[i]=n_max-((n_max-n_min)/(num-1))*(i-1);
    //else
    kernel[i] = n_max/pow(2,(i-1)/2);
    //kernel[i] = n_max - floor(i/2)*n_max/(bit_num-1);
    kernel[i+1] = -kernel[i];
  }
  //uncomment this if not the first iteration here2
	/*std::ifstream fi("/home/xuyuhui/k_0.txt");
        std::ofstream fo;	
	fo.open("/home/xuyuhui/tmp.txt");
	int n = 0;
	double k;
	while (fi>>k) {
		if (n < num) {
			kernel[n] = k;
		}
		else {
			fo << k << std::endl;
		}
		n++;
	}
	fi.close();
	fo.close();
	fi.open("/home/xuyuhui/tmp.txt");
	fo.open("/home/xuyuhui/k_0.txt");
	while (fi>>k) {
		fo << k << std::endl;
	}
	fi.close();
	fo.close();*/

  //srand(time(0));
  //kernel[9] = 0.1;
  //for(int i=2*bit_num-1; i<num; i++) kernel[i] = 2.0*n_max*rand()/INT_MAX-n_max;
  // Kmeans
  for(int e=0; e<epoch; e++){
    kmeans_pre(data_vec, T_vec, kernel, num, count);
  }
  for(int i=0;i<num;i++) kernel_num[i]=0;
  for(int i=0; i<count; i++){
    kernel_num[int(T_vec[i])] +=1;
  }

  for(int i=0; i<count; i++){
    dis_vec[i] = 1.0;
  }

  for(int i=0; i<count; i++){
    if(T_vec[i]<5){//here 3
      data_vec[i] = kernel[int(T_vec[i])];
      dis_vec[i] = 0;
    }
    else dis_vec[i] = 1;
  }

  std::ofstream fout;
  fout.open("/home/xuyuhui/k_0.txt",ios::app);
  for(int i=0;i<num;i++) fout<<kernel[i]<<std::endl;//<<kernel_num[i]<<std::endl;
  fout.close();
  
}

} //namespace
