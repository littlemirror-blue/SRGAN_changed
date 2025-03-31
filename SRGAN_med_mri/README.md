# SRGAN_med_MRI
A PyTorch implementation of SRGAN based on CVPR 2017 paper , and fpr MRI image super resolution.
[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802).

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- PyTorch
```
conda install pytorch torchvision -c pytorch
```
- opencv
```
conda install opencv
```

## Datasets

### Train、Val Dataset | Test Image Dataset
all are sampled from [7 Tesla Magnetic Resonance Imaging]( https://www.nature.com/articles/s41597-021-00923-w#Sec12)


## Usage

### Train
```
python train.py

optional arguments:
--crop_size                   training images crop size [default value is 88]
--upscale_factor              super resolution upscale factor [default value is 4](choices:[2, 4, 8])
--num_epochs                  train epoch number [default value is 100]
```
The output val super resolution images are on `training_results` directory.

### Test Benchmark Datasets
```
python test_benchmark.py

optional arguments:
--upscale_factor              super resolution upscale factor [default value is 4]
--model_name                  generator model epoch name [default value is netG_epoch_4_100.pth]
```
The output super resolution images are on `benchmark_results` directory.

### Test Single Image
```
python test_image.py

optional arguments:
--upscale_factor              super resolution upscale factor [default value is 4]
--test_mode                   using GPU or CPU [default value is 'GPU'](choices:['GPU', 'CPU'])
--image_name                  test low resolution image name
--model_name                  generator model epoch name [default value is netG_epoch_4_100.pth]
```
The output super resolution image are on the same directory.


## Benchmarks
**Upscale Factor = 2**

on a NVIDIA RTX 4060 GPU. 

> Image Results

DataSet,psnr,ssim
run-01_offline,33.021878614270875,0.7925928201940324
run-02_biasCorrected,33.62438808927625,0.8988154599691007
run-02_offline,33.742996203680384,0.8865199305872986


**Upscale Factor = 4**



> Image Results

DataSet,psnr,ssim
run-01_offline,31.10224022569401,0.8257219272631185
run-02_biasCorrected,29.60462313440357,0.7138994122527972
run-02_offline,30.772513472849184,0.7961333575456039


**Upscale Factor = 8**

> Image Results

DataSet,psnr,ssim
run-01_offline,28.368426917211956,0.6271169119411044
run-02_biasCorrected,25.6459050520094,0.40064555874456725
run-02_offline,27.912305133551722,0.5522248461626578



