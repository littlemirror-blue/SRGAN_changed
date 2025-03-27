import argparse
import os
from math import log10
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model import Generator

def extract_key_from_filename(filename):
    """从文件名中提取分类键：run-X_T1w_<processing>"""
    parts = filename.split('_')
    # 提取run编号(parts[2])、模态(parts[3])和处理方式(parts[4])
    return f"{parts[2]}_{parts[3]}_{parts[4]}"

def main():
    parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
    # 默认修改为epoch_4_22.pth(G为生成器,D为判别器(只在训练时使用))
    #parser.add_argument('--model_name', default='netG_epoch_4_22.pth', type=str, help='generator model epoch name')
    # 放大修改为默认为8
    parser.add_argument('--upscale_factor', default=8, type=int, help='super resolution upscale factor')
    # 默认修改为epoch_8_98.pth(G为生成器,D为判别器(只在训练时使用))
    parser.add_argument('--model_name', default='netG_epoch_8_98.pth', type=str, help='generator model epoch name')
    opt = parser.parse_args()
    UPSCALE_FACTOR = opt.upscale_factor
    MODEL_NAME = opt.model_name

    # 预定义results字典，包含已知的处理类型
    results = {
        'run-02_T1w_offline_manual_withRef': {'psnr': [], 'ssim': []},
        'run-02_T1w_biasCorrected': {'psnr': [], 'ssim': []},
        'run-01_T1w_offline_manual_withRef': {'psnr': [], 'ssim': []}
    }

    model = Generator(UPSCALE_FACTOR).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, weights_only=True))

    test_set = TestDatasetFromFolder('data/test', upscale_factor=UPSCALE_FACTOR)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

    out_path = 'benchmark_results/SRF_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for image_name, lr_image, hr_restore_img, hr_image in test_bar:
        image_name = image_name[0]
        with torch.no_grad():
            lr_image = lr_image
            hr_image = hr_image

        if torch.cuda.is_available():
            lr_image = lr_image.cuda()
            hr_image = hr_image.cuda()

        sr_image = model(lr_image)

        if hr_image.size() != sr_image.size():
            print(f"Adjusting sr_image size from {sr_image.size()} to {hr_image.size()}")
            sr_image = F.interpolate(sr_image, size=(hr_image.size(2), hr_image.size(3)), mode='bicubic', align_corners=False)

        mse = ((hr_image - sr_image) ** 2).data.mean()
        psnr = 10 * log10(1 / mse)
        ssim = pytorch_ssim.ssim(sr_image, hr_image).item()

        test_images = torch.stack(
            [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.data.cpu().squeeze(0)),
             display_transform()(sr_image.data.cpu().squeeze(0))])
        image = utils.make_grid(test_images, nrow=3, padding=5)
        utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) +
                         image_name.split('.')[-1], padding=5)

        # 获取分类键
        key = extract_key_from_filename(image_name)
        
        # 保存结果到对应的类别
        if key in results:
            results[key]['psnr'].append(psnr)
            results[key]['ssim'].append(ssim)
        else:
            print(f"Warning: Unexpected file type - {image_name}")

    # 保存统计结果
    out_path = 'statistics/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    saved_results = {'psnr': [], 'ssim': []}
    index_labels = []
    
    for key, metrics in results.items():
        psnr = np.array(metrics['psnr'])
        ssim = np.array(metrics['ssim'])
        
        if len(psnr) == 0 or len(ssim) == 0:
            psnr_mean = 'No data'
            ssim_mean = 'No data'
        else:
            psnr_mean = psnr.mean()
            ssim_mean = ssim.mean()
        
        saved_results['psnr'].append(psnr_mean)
        saved_results['ssim'].append(ssim_mean)
        index_labels.append(key)

    data_frame = pd.DataFrame(saved_results, index=index_labels)
    data_frame.index.name = 'DatasetType'
    data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_test_results.csv')

if __name__ == '__main__':
    main()