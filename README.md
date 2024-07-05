# WeConvene-Learned-Image-Compression-with-Wavelet-Domain-Convolution-and-Entropy-Model
WeConvene: Learned Image Compression with  Wavelet-Domain Convolution and Entropy Model



This repository contains the code for reproducing the results with trained models, in the following paper:

Our code is based on the paper named Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules. [arXiv](https://arxiv.org/abs/2001.01568), CVPR2020. Zhengxue Cheng, Heming Sun, Masaru Takeuchi, Jiro Katto

Our paper is Learned Image Compression with Discretized Gaussian-Laplacian-Logistic Mixture Model and Concatenated Residual Modules. [arXiv](https://arxiv.org/abs/2107.06463).
Haisheng Fu, Feng Liang, Jianping Lin, Bing Li, Mohammad Akbari, Jie Liang, Guohe Zhang, Dong Liu, Chengjie Tu, Jingning Han


## Paper Summary

Recently deep learning-based image compression methods have achieved significant achievements and gradually outperformed traditional approaches including the latest standard Versatile Video Coding (VVC) in both PSNR and MS-SSIM metrics. Two key components of learned image compression frameworks are the entropy model of the latent representations
and the encoding/decoding network architectures. Various models have been proposed, such as autoregressive, softmax, logistic mixture, Gaussian mixture, and Laplacian. Existing schemes only use one of these models. However, due to the vast diversity of images, it is not optimal to use one model for all images, even different regions of one image. In this paper, we propose a more flexible discretized Gaussian-Laplacian-Logistic mixture model (GLLMM) for the latent representations, which can adapt to different contents in different images and different regions of one image more accurately. Besides, in the encoding/decoding network design part, we propose a concatenated residual blocks
(CRB), where multiple residual blocks are serially connected with additional shortcut connections. The CRB can improve the learning ability of the network, which can further improve the compression performance. Experimental results using the Kodak and Tecnick datasets show that the proposed scheme outperforms all the state-of-the-art learning-based methods and existing compression standards including VVC intra coding (4:4:4 and 4:2:0) in terms of the PSNR and MS-SSIM.

### Environment 

* Python==3.6.4

* Tensorflow==1.14.0

* [RangeCoder](https://github.com/lucastheis/rangecoder)

```   
    pip3 install range-coder
```

* [Tensorflow-Compression](https://github.com/tensorflow/compression) ==1.2

```
    pip3 install tensorflow-compression or 
    pip3 install tensorflow_compression-1.2-cp36-cp36m-manylinux1_x86_64.whl
```

### Test Usage

* Download the pre-trained [models](https://pan.baidu.com/s/1VZ8EZZzX8VKJg4auKxVytQ) (The Extraction code is i6p3. These models are optimized by PSNR using lambda = 0.0016(number filters=128)) and lambda = 0.03(number filters=256)).

* Run the following py files can encode or decode the input file. 

```
   python Encoder_Decoder_cvpr_blocks_leaky_GLLMM_directly_bits_github.py
   note that:
   endcoder_main(); // the Encoder code
   decoder_main();  // the Decoder  code
   path ='xxx';     // the test image 
   save_image_name_path=''; // save the bit stream files.
   num_filters = 128 or 256;  // 128 for low bit rates and 256 for high bit rates.
   
```



## Reconstructed Samples

Comparisons of reconstructed samples are given in the following.

![](https://github.com/fengyurenpingsheng/Learned-image-compression-with-GLLMM/blob/main/Figure/example.png)


## Evaluation Results

![](https://github.com/fengyurenpingsheng/Learned-image-compression-with-GLLMM/blob/main/Figure/result.png)

## Notes


If you think it is useful for your reseach, please cite our paper. 
```

@ARTICLE{Fu_2023,
  author={Fu, Haisheng and Liang, Feng and Lin, Jianping and Li, Bing and Akbari, Mohammad and Liang, Jie and Zhang, Guohe and Liu, Dong and Tu, Chengjie and Han, Jingning},
  journal={IEEE Transactions on Image Processing}, 
  title={Learned Image Compression With Gaussian-Laplacian-Logistic Mixture Model and Concatenated Residual Modules}, 
  year={2023},
  volume={32},
  number={},
  pages={2063-2076},
 }

```

