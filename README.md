# SLQ
Caffe implementation of paper ["Deep Neural Network Compression with Single and Multiple Level Quantization"](https://arxiv.org/pdf/1803.03289.pdf)<br>

## How to use
make the caffe as usual<br>

Download the pretrained AlexNet caffemodel. <br>
In models/bvlc_alexnet fintune alexnet you can have the first quantization iteration of SLQ. <br>

The key code is in the src/caffe/blob.cpp and include/caffe/kmeans.hpp. To have the other iteration you can change the code in kmeans.hpp and finetune the caffemodel of the first iteration.<br>

## Details
Three marks "here1","here2","here3"means three palces need to change in the second iteration.
#### here1
Here determines how many kernels need to update by the means of the weights, we do not update the quantized kernels:
During the 5-bit quantization(17 kernels) of Alexnet, 5 iterations, the changes are i=1;i=5;i=9;i=13;i=15 respectively:
```
  for(int i=1; i<num; i++){//here1
  if(kernel_num[i]>0) kernel[i] = kernel_new[i] / kernel_num[i];}
}
```

#### here2
Here is to read the kernel of the last quantization iteration, only the first iteration need to comment this section.
```
/*std::ifstream fi("/home/xuyuhui/k_0.txt");
        std::ofstream fo;	
        
```
#### here3
Here is to partition the clusters and quantize and fix the weights in the chosen clusters.During the 5-bit quantization(17 kernels) of Alexnet, 5 iterations, the changes are T_vec[i]<5;T_vec[i]<9;T_vec[i]<13;T_vec[i]<15;T_vec[i]<17 respectively:
```
  for(int i=0; i<count; i++){
    if(T_vec[i]<5){//here 3
      data_vec[i] = kernel[int(T_vec[i])];
      dis_vec[i] = 0;
    }
    else dis_vec[i] = 1;
  }
```


### Acceleration
The code of Low-rank decomposition method [Trained Rank Pruning](https://github.com/yuhuixu1993/Trained-Rank-Pruning) is also available. 

## Citation
```latex
@article{xu2018deep,
  title={Deep Neural Network Compression with Single and Multiple Level Quantization},
  author={Xu, Yuhui and Wang, Yongzhuang and Zhou, Aojun and Lin, Weiyao and Xiong, Hongkai},
  journal={arXiv preprint arXiv:1803.03289},
  year={2018}
}
