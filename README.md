# WeConvene-Learned-Image-Compression-with-Wavelet-Domain-Convolution-and-Entropy-Model
WeConvene: Learned Image Compression with  Wavelet-Domain Convolution and Entropy Model



This repository contains the code for reproducing the results with trained models, in the following paper:

Our code is based on the paper named Learned Image Compression with Mixed Transformer-CNN Architectures. [arXiv](https://arxiv.org/abs/2303.14978), CVPR2023. Jinming Liu, Heming Sun, Jiro Katto.

Our paper is WeConvene: Learned Image Compression with  Wavelet-Domain Convolution and Entropy Model. [arXiv](https://arxiv.org/abs/2107.06463).
Haisheng Fu, Feng Liang, Jianping Lin, Bing Li, Mohammad Akbari, Jie Liang, Guohe Zhang, Dong Liu, Chengjie Tu, Jingning Han


## Paper Summary

Recently learned image compression (LIC) has achieved great progress and even outperformed the traditional approaches. However, LIC mainly reduces spatial redundancy in the autoencoder networks and entropy coding, but has not fully removed the frequency-domain correlation explicitly via linear transform (such as DCT or wavelet transform), which is the cornerstone of the traditional methods. To address this critical limitation, in this paper, we propose a surprisingly simple but efficient framework, which introduces the discrete wavelet transform (DWT) to both the convolution layers and entropy coding of LIC. First, in both the core and hyperprior autoencoder networks, we propose a Wavelet-domain Convolution (WeConv) module at selected layers to reduce the frequency-domain correlation explicitly and make the signal sparser. Experimental results show that by using the simplest Harr wavelet transform, WeConv can already achieve 0.2-0.25 dB gain in the rate-distortion (R-D) performance with negligible change of model size and running time. We also perform entropy coding and quantization in the wavelet domain, and propose a Wavelet-domain Channel-wise Auto-Regressive entropy Model (WeChARM), where the latent representations are quantized and entropy coded in the wavelet domain instead of spatial domain. Moreover, the entropy coding is split into two steps. We first encode and decode all the low-frequency wavelet transform coefficients, and then use them as prior information to encode and decode the high-frequency coefficients. The channel-wise entropy coding is further used in each step. WeChARM can further improve the R-D performance by 0.25-0.3 dB, with moderate increase of model size and running time. By combining WeConv and WeChARM, the proposed WeConvene scheme achieves superior R-D performance compared to other state-of-the-art LIC methods as well as the latest H.266/VVC. In particular, it achieves a BD-rate reduction of 9.11%, 9.46%, and 9.20% over H.266/VVC on the Kodak, Tecnick, and CLIC datasets, respectively. Better performance can be achieved by using more advanced wavelet transforms. The proposed convolution-based system is also easier to train and has less requirements on GPU than transformer-based schemes.

### Environment 

* Python==3.10.0

* Compressai==1.2.6


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

