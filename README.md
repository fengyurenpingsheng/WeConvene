# WeConvene-Learned-Image-Compression-with-Wavelet-Domain-Convolution-and-Entropy-Model
This repository contains the code for reproducing the results with trained models for the following ECCV2024 paper:

WeConvene: Learned Image Compression with Wavelet-Domain Convolution and Entropy Model. [arXiv](https://arxiv.org/abs/2407.09983), ECCV2024. Haisheng Fu and Jie Liang and Zhenman Fang and Jingning Han and Feng Liang and Guohe Zhang.

Our code is based on the paper named Learned Image Compression with Mixed Transformer-CNN Architectures. [arXiv](https://arxiv.org/abs/2303.14978), CVPR2023. Jinming Liu, Heming Sun, Jiro Katto.



## Paper Summary

Recently learned image compression (LIC) has achieved great progress and even outperformed the traditional approaches. However, LIC mainly reduces spatial redundancy in the autoencoder networks and entropy coding, but has not fully removed the frequency-domain correlation explicitly via linear transform (such as DCT or wavelet transform), which is the cornerstone of the traditional methods. To address this critical limitation, in this paper, we propose a surprisingly simple but efficient framework, which introduces the discrete wavelet transform (DWT) to both the convolution layers and entropy coding of LIC. First, in both the core and hyperprior autoencoder networks, we propose a Wavelet-domain Convolution (WeConv) module at selected layers to reduce the frequency-domain correlation explicitly and make the signal sparser. Experimental results show that by using the simplest Harr wavelet transform, WeConv can already achieve 0.2-0.25 dB gain in the rate-distortion (R-D) performance with negligible change of model size and running time. We also perform entropy coding and quantization in the wavelet domain, and propose a Wavelet-domain Channel-wise Auto-Regressive entropy Model (WeChARM), where the latent representations are quantized and entropy coded in the wavelet domain instead of spatial domain. Moreover, the entropy coding is split into two steps. We first encode and decode all the low-frequency wavelet transform coefficients, and then use them as prior information to encode and decode the high-frequency coefficients. The channel-wise entropy coding is further used in each step. WeChARM can further improve the R-D performance by 0.25-0.3 dB, with moderate increase of model size and running time. By combining WeConv and WeChARM, the proposed WeConvene scheme achieves superior R-D performance compared to other state-of-the-art LIC methods as well as the latest H.266/VVC. In particular, it achieves a BD-rate reduction of 9.11%, 9.46%, and 9.20% over H.266/VVC on the Kodak, Tecnick, and CLIC datasets, respectively. Better performance can be achieved by using more advanced wavelet transforms. The proposed convolution-based system is also easier to train and has less requirements on GPU than transformer-based schemes.

### Environment 

* Python==3.10.0

* Compressai==1.2.6

## Training
``` 
CUDA_VISIBLE_DEVICES='0' python -u ./train.py -d [path of training dataset] \
    --cuda --N 128 --lambda 0.05 --epochs 201 --lr_epoch 130 180 \
    --save_path [path for checkpoint] --save \
    --checkpoint [path of the pretrained checkpoint]
```


### Test Usage

```

python eval.py --checkpoint [path of the pretrained checkpoint] --data [path of testing dataset] --cuda --real
   
```


## Pretrained Model
| Lambda | Metric | Link |
|--------|--------|------|
| 0.0025   | MSE    |   [link](https://pan.baidu.com/s/1VaD4yzY0mwbcWp1XKERnqw?pwd=0ih5)   |
| 0.0035   | MSE    |   [link](https://pan.baidu.com/s/1VaD4yzY0mwbcWp1XKERnqw?pwd=0ih5)   |
| 0.0067   | MSE    |   [link](https://pan.baidu.com/s/1VaD4yzY0mwbcWp1XKERnqw?pwd=0ih5)   |
| 0.013    | MSE    |   [link](https://pan.baidu.com/s/1VaD4yzY0mwbcWp1XKERnqw?pwd=0ih5)   |
| 0.025    | MSE    |   [link](https://pan.baidu.com/s/1VaD4yzY0mwbcWp1XKERnqw?pwd=0ih5)   |
| 0.05     | MSE    |   [link](https://pan.baidu.com/s/1VaD4yzY0mwbcWp1XKERnqw?pwd=0ih5)   |



## Citation
``` 
@misc{Fu2024ECCV2024,
      title={WeConvene: Learned Image Compression with Wavelet-Domain Convolution and Entropy Model}, 
      author={Haisheng Fu and Jie Liang and Zhenman Fang and Jingning Han and Feng Liang and Guohe Zhang},
      year={2024},
      eprint={2407.09983},
      archivePrefix={arXiv},
      primaryClass={stat.AP},
      url={https://arxiv.org/abs/2407.09983}, 
}
```



