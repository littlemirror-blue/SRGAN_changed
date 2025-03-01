import os
import nibabel as nib
import numpy as np
from PIL import Image
import random
import shutil

# 定义目录路径
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

# 创建所有必要的目录
for dir_path in [all_slices_dir, train_dir, val_dir, test_srf8_target_dir, test_srf8_data_dir,
                 test_srf4_target_dir, test_srf4_data_dir, test_srf2_target_dir, test_srf2_data_dir]:
    os.makedirs(dir_path, exist_ok=True)

# 定义 resize 函数，与 data_SRF.py 一致
def resize_image(image, max_size=1024):
    """将图像分辨率限制在 max_size x max_size 以内，保持宽高比"""
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        image = image.resize((new_width, new_height), Image.BICUBIC)
    return image

# 步骤 1：将 .nii 文件转换为 .png 切片，并限制最大尺寸
for nii_file in os.listdir(nii_dir):
    if nii_file.endswith('.nii'):
        # 加载 .nii 文件
        img = nib.load(os.path.join(nii_dir, nii_file))
        data = img.get_fdata()
        # 提取所有轴向切片
        for i in range(data.shape[2]):
            slice_data = data[:, :, i]
            # 归一化到 0-255 范围
            if slice_data.max() > slice_data.min():  # 防止除以零
                slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255
            else:
                slice_data = slice_data - slice_data.min()  # 如果最大最小值相等，直接平移
            slice_data = slice_data.astype(np.uint8)
            # 转换为 PIL 图像并限制最大尺寸
            slice_img = Image.fromarray(slice_data).convert('L')
            slice_img = resize_image(slice_img, max_size=1024)
            # 保存切片
            slice_filename = f"{nii_file[:-4]}_slice_{i:03d}.png"
            slice_img.save(os.path.join(all_slices_dir, slice_filename))

# 步骤 2：列出所有切片并随机分配
all_slices = [f for f in os.listdir(all_slices_dir) if f.endswith('.png')]
random.shuffle(all_slices)

# 确保有足够的切片
if len(all_slices) < 4000:
    raise ValueError(f"切片总数 {len(all_slices)} 小于所需数量 4000，请检查 .nii 文件数量或切片提取逻辑")

# 分配到 train、val 和 test
train_slices = all_slices[:1800]
val_slices = all_slices[1800:2200]  # 1800 + 400
test_slices = all_slices[2200:4000]  # 2200 + 1800

# 移动文件到对应目录
for slice_file in train_slices:
    shutil.move(os.path.join(all_slices_dir, slice_file), os.path.join(train_dir, slice_file))
for slice_file in val_slices:
    shutil.move(os.path.join(all_slices_dir, slice_file), os.path.join(val_dir, slice_file))
for slice_file in test_slices:
    shutil.move(os.path.join(all_slices_dir, slice_file), os.path.join(test_srf8_target_dir, slice_file))

# 步骤 3：为测试集生成 LR 和不同 SRF 的 Target 图像
for hr_file in os.listdir(test_srf8_target_dir):
    hr_path = os.path.join(test_srf8_target_dir, hr_file)
    hr_img = Image.open(hr_path)
    width, height = hr_img.size

    # SRF_8: 生成 LR，下采样 8 倍
    lr_width = width // 8
    lr_height = height // 8
    if lr_width > 0 and lr_height > 0:
        lr_img = hr_img.resize((lr_width, lr_height), Image.BICUBIC)
        lr_img.save(os.path.join(test_srf8_data_dir, hr_file))
    else:
        print(f"警告：{hr_file} 尺寸过小，无法下采样 8 倍，已跳过")

    # SRF_4: 生成 Target，下采样 2 倍
    srf4_target_width = width // 2
    srf4_target_height = height // 2
    if srf4_target_width > 0 and srf4_target_height > 0:
        srf4_target_img = hr_img.resize((srf4_target_width, srf4_target_height), Image.BICUBIC)
        srf4_target_img.save(os.path.join(test_srf4_target_dir, hr_file))
    else:
        print(f"警告：{hr_file} 尺寸过小，无法下采样 2 倍，已跳过")

    # SRF_2: 生成 Target，下采样 4 倍
    srf2_target_width = width // 4
    srf2_target_height = height // 4
    if srf2_target_width > 0 and srf2_target_height > 0:
        srf2_target_img = hr_img.resize((srf2_target_width, srf2_target_height), Image.BICUBIC)
        srf2_target_img.save(os.path.join(test_srf2_target_dir, hr_file))
    else:
        print(f"警告：{hr_file} 尺寸过小，无法下采样 4 倍，已跳过")

# 步骤 4：将 SRF_8 的 LR 图像复制到 SRF_4 和 SRF_2 的 data 文件夹
for lr_file in os.listdir(test_srf8_data_dir):
    src_path = os.path.join(test_srf8_data_dir, lr_file)
    shutil.copy(src_path, os.path.join(test_srf4_data_dir, lr_file))
    shutil.copy(src_path, os.path.join(test_srf2_data_dir, lr_file))

print("处理完成，所有图像已生成并分配到指定目录。")