# SRGAN_med_xray
A PyTorch implementation of SRGAN based on CVPR 2017 paper , and for xray image super resolution.
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
all are sampled from [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)


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
IM,31.496432261974036,0.8664694177931633
NORMAL2,31.089157376965158,0.8640953775608178


**Upscale Factor = 4**



> Image Results

DataSet,psnr,ssim
IM,31.294259135954395,0.8127471925555796
NORMAL2,30.870179874763522,0.8242938876152038


**Upscale Factor = 8**
on NVIDIA  GPU T4 x2

> Image Results

DataSet,psnr,ssim
IM,28.948336811568637,0.8055744205695995
NORMAL2,28.906481468841143,0.8249477928335016



