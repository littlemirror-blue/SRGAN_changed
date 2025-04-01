# Medical Super-Resolution GAN Projects

This repository contains a collection of Super-Resolution Generative Adversarial Network (SRGAN) implementations focused on medical imaging applications. The projects are built upon the original SRGAN architecture with modifications tailored for different medical imaging modalities.

## Projects Overview

- **SRGAN-med**: General medical image super-resolution implementation(xray led png image)[readme](./SRGAN-med/README.md)
- **SRGAN_med_mri**: SRGAN implementation specifically optimized for MRI images [readme](./SRGAN_med_mri/README.md)
- **SRGAN_med_CT**: SRGAN implementation specifically optimized for CT images [readme](./SRGAN_med_CT/README.md)

Each project adapts the SRGAN architecture to address the unique characteristics and challenges of different medical imaging modalities.

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- PIL (Pillow)
- numpy
- matplotlib

You can install the required packages using:

```bash
pip install torch torchvision pillow numpy matplotlib
```

## Project Structure

Each project follows a similar structure:

- `model.py`: Contains the SRGAN model architecture
- `data_utils.py`: Utilities for data loading and preprocessing
- `train.py`: Training script for the model
- `test.py`: Testing and evaluation script
- `pytorch_ssim`: Implementation of the Structural Similarity Index (SSIM) for PyTorch

## Usage

### Training

To train a model, navigate to the specific project directory and run:

```bash
python train.py --train_image_dir [TRAINING_IMAGES_PATH] --upscale_factor [FACTOR] --num_epochs [EPOCHS]
```

### Testing

To test a trained model:

```bash
python test.py --test_image_dir [TEST_IMAGES_PATH] --upscale_factor [FACTOR] --model_name [MODEL_PATH]
```

## Model Architecture

The SRGAN architecture consists of:

1. **Generator**: A deep residual network with upsampling blocks
   - Uses convolutional layers followed by PReLU activation
   - Employs PixelShuffle for efficient upsampling

2. **Discriminator**: A convolutional neural network
   - Distinguishes between super-resolved images and high-resolution ground truth

3. **Loss Functions**:
   - Perceptual loss (based on VGG features)
   - Adversarial loss
   - Content loss (MSE)
   - SSIM for evaluation

## Medical Imaging Adaptations

### MRI-Specific Adaptations (SRGAN_med_mri)
- Modified network parameters to handle MRI-specific noise characteristics
- Adjusted training process for grayscale images
- Custom preprocessing for MRI data

### CT-Specific Adaptations (SRGAN_med_CT)
- Optimized for CT image characteristics
- Handles different contrast ranges
- Specific data augmentation techniques for CT images

## Evaluation Metrics

The models are evaluated using:
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)
- Visual assessment of super-resolved images

## Results

Each project directory contains sample results demonstrating the performance of the models on different medical imaging modalities.

## Acknowledgements

This work builds upon the original SRGAN implementation. The base SRGAN architecture is referenced from the SRGAN-master project.





