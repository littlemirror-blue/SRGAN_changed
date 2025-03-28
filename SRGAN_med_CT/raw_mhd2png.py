import SimpleITK as sitk
import numpy as np
from PIL import Image
import os
import glob

# 定义存储路径
save_dir = r'C:\Users\13550\Downloads\SRGAN_changed\SRGAN_med_CT\data\all_slices'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 获取所有 .mhd 文件
mhd_files = glob.glob(r'C:\Users\13550\Downloads\SRGAN_changed\SRGAN_med_CT\data\subset0\*.mhd')

# 处理每个 .mhd 文件
for mhd_file in mhd_files:
    # 读取图像
    image = sitk.ReadImage(mhd_file)
    
    # 将图像转换为 NumPy 数组，形状为 (depth, height, width)
    array = sitk.GetArrayFromImage(image)
    depth = array.shape[0]
    
    # 获取文件名中的前缀（患者ID）
    base_name = os.path.basename(mhd_file).replace('.mhd', '')
    
    # 遍历每个切片
    for i in range(depth):
        slice_data = array[i, :, :]
        
        # 检查切片中 HU 值在 -1000 到 -400 之间的像素比例（肺部组织范围）
        mask = (slice_data >= -1000) & (slice_data <= -400)
        ratio = np.sum(mask) / slice_data.size
        
        # 如果比例大于 5%，认为切片包含肺部组织
        if ratio > 0.05:
            # 应用窗宽窗位调整
            window_width = 1500  # 窗宽
            window_level = -600  # 窗位
            min_HU = window_level - window_width / 2  # 最小 HU 值
            max_HU = window_level + window_width / 2  # 最大 HU 值
            
            # 裁剪并归一化到 0-255
            slice_data_clipped = np.clip(slice_data, min_HU, max_HU)
            slice_data_normalized = (slice_data_clipped - min_HU) / (max_HU - min_HU) * 255
            slice_data_uint8 = slice_data_normalized.astype(np.uint8)
            
            # 保存为 .png 文件
            slice_name = f'{base_name}_slice_{i:04d}.png'
            save_path = os.path.join(save_dir, slice_name)
            Image.fromarray(slice_data_uint8).save(save_path)

print("所有有效切片已保存至", save_dir)