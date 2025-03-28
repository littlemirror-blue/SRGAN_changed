# ====================== ↓ nii2png较为准确，SRF_VOC分发有问题 ======================
import os
import nibabel as nib
import numpy as np
from PIL import Image
import random
import shutil
from nilearn.masking import compute_brain_mask  # 关键：Nilearn的脑掩码计算
from scipy.ndimage import binary_closing  # 可选：掩码后处理

# ====================== 目录定义 ======================
nii_dir = 'data/nii'
all_slices_dir = 'data/all_slices'
train_dir = 'data/VOC2012/train'
val_dir = 'data/VOC2012/val'
test_srf8_target_dir = 'data/test/SRF_8/target'
test_srf8_data_dir = 'data/test/SRF_8/data'
test_srf4_target_dir = 'data/test/SRF_4/target'
test_srf4_data_dir = 'data/test/SRF_4/data'
test_srf2_target_dir = 'data/test/SRF_2/target'
test_srf2_data_dir = 'data/test/SRF_2/data'

# ====================== 创建目录 ======================
for dir_path in [all_slices_dir, train_dir, val_dir, 
                 test_srf8_target_dir, test_srf8_data_dir,
                 test_srf4_target_dir, test_srf4_data_dir,
                 test_srf2_target_dir, test_srf2_data_dir]:
    os.makedirs(dir_path, exist_ok=True)

# ====================== 核心函数 ======================
def resize_image(image, max_size=1024):
    """限制图像最大尺寸，保持宽高比"""
    width, height = image.size
    if max(width, height) > max_size:
        ratio = max_size / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        image = image.resize(new_size, Image.BICUBIC)
    return image

def generate_brain_mask(nii_path, output_mask_path, threshold=0.7):
    """
    使用Nilearn生成脑掩码
    参数:
        threshold: 控制掩码松紧 (0-1)，默认0.5
    """
    mask = compute_brain_mask(nii_path, threshold=threshold)
    
    # 可选：形态学后处理（填充空洞）
    mask_data = mask.get_fdata()
    mask_data = binary_closing(mask_data, structure=np.ones((3, 3, 3)))
    mask = nib.Nifti1Image(mask_data.astype(np.uint8), mask.affine)
    
    mask.to_filename(output_mask_path)
    return output_mask_path

def get_valid_z_range(mask_path):
    """从脑掩码中提取有效Z轴范围"""
    mask = nib.load(mask_path)
    mask_data = mask.get_fdata()
    z_valid = np.any(mask_data > 0, axis=(0, 1))  # Z轴上是否有脑组织
    z_indices = np.where(z_valid)[0]
    return z_indices.min(), z_indices.max() if len(z_indices) > 0 else (0, 0)

# ====================== 主流程 ======================
# 步骤1：处理所有.nii文件，生成有效切片
for nii_file in os.listdir(nii_dir):
    if nii_file.endswith('.nii'):
        nii_path = os.path.join(nii_dir, nii_file)
        print(f"正在处理: {nii_file}")
        
        # 生成脑掩码
        mask_path = os.path.join(nii_dir, f"{nii_file[:-4]}_mask.nii.gz")
        generate_brain_mask(nii_path, mask_path, threshold=0.5)
        
        # 获取有效Z轴范围
        z_min, z_max = get_valid_z_range(mask_path)
        print(f"有效切片范围: Z={z_min}~{z_max}")
        
        # 加载原始图像
        img = nib.load(nii_path)
        data = img.get_fdata()
        
        # 仅保存有效范围内的切片
        for z in range(z_min, z_max + 1):
            slice_data = data[:, :, z]
            
            # 归一化到0-255
            if np.ptp(slice_data) > 0:  # 防止除以零
                slice_data = (slice_data - slice_data.min()) / np.ptp(slice_data) * 255
            slice_data = slice_data.astype(np.uint8)
            
            # 转换为图像并调整尺寸
            slice_img = Image.fromarray(slice_data).convert('L')
            slice_img = resize_image(slice_img)
            
            # 保存切片
            slice_name = f"{nii_file[:-4]}_slice_{z:03d}.png"
            slice_img.save(os.path.join(all_slices_dir, slice_name))

# 步骤2：随机分配切片到训练/验证/测试集
all_slices = [f for f in os.listdir(all_slices_dir) if f.endswith('.png')]
random.shuffle(all_slices)

if len(all_slices) < 4000:
    raise ValueError(f"切片不足4000张（当前{len(all_slices)}），请检查输入数据")

# 分配比例：1800训练 / 400验证 / 1800测试
train_slices = all_slices[:1800]
val_slices = all_slices[1800:2200]
test_slices = all_slices[2200:4000]

# 移动文件到对应目录
def move_files(file_list, src_dir, dst_dir):
    for f in file_list:
        shutil.move(os.path.join(src_dir, f), os.path.join(dst_dir, f))

move_files(train_slices, all_slices_dir, train_dir)
move_files(val_slices, all_slices_dir, val_dir)
move_files(test_slices, all_slices_dir, test_srf8_target_dir)

# 步骤3：生成测试集的下采样版本
for hr_file in os.listdir(test_srf8_target_dir):
    hr_img = Image.open(os.path.join(test_srf8_target_dir, hr_file))
    
    # SRF_8 (1/8)
    lr_img = hr_img.resize((hr_img.width // 8, hr_img.height // 8), Image.BICUBIC)
    lr_img.save(os.path.join(test_srf8_data_dir, hr_file))
    
    # SRF_4 (1/2)
    srf4_img = hr_img.resize((hr_img.width // 2, hr_img.height // 2), Image.BICUBIC)
    srf4_img.save(os.path.join(test_srf4_target_dir, hr_file))
    
    # SRF_2 (1/4)
    srf2_img = hr_img.resize((hr_img.width // 4, hr_img.height // 4), Image.BICUBIC)
    srf2_img.save(os.path.join(test_srf2_target_dir, hr_file))

# 步骤4：复制LR图像到其他测试目录
for lr_file in os.listdir(test_srf8_data_dir):
    for target_dir in [test_srf4_data_dir, test_srf2_data_dir]:
        shutil.copy(
            os.path.join(test_srf8_data_dir, lr_file),
            os.path.join(target_dir, lr_file)
        )

print("处理完成！")