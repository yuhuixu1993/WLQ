# SLQ
Caffe implementation of paper "Deep Neural Network Compression with Single and Multiple Level Quantization"<https://arxiv.org/pdf/1803.03289.pdf><br>

### How to use
make the caffe as usual<br>

Download the pretrained AlexNet caffemodel. <br>
In models/bvlc_alexnet fintune alexnet you can have the first quantization iteration of SLQ. <br>

The key code is in the src/caffe/blob.cpp and include/caffe/kmeans.hpp. To have the other iteration you can change the code in kmeans.hpp and finetune the caffemodel of the first iteration.

### Citation
```latex
@article{xu2018deep,
  title={Deep Neural Network Compression with Single and Multiple Level Quantization},
  author={Xu, Yuhui and Wang, Yongzhuang and Zhou, Aojun and Lin, Weiyao and Xiong, Hongkai},
  journal={arXiv preprint arXiv:1803.03289},
  year={2018}
}
