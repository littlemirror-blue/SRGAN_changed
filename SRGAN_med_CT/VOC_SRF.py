import os
from PIL import Image

def is_image_file(filename):
    """检查文件是否为图像文件"""
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

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

def generate_lr_image(hr_image, upscale_factor):
    """从HR图像生成LR图像"""
    lr_width = hr_image.width // upscale_factor
    lr_height = hr_image.height // upscale_factor
    if lr_width > 0 and lr_height > 0:
        lr_image = hr_image.resize((lr_width, lr_height), Image.BICUBIC)
        return lr_image
    else:
        return None

def process_and_distribute_images(original_dir, data_dir):
    """
    处理并分发图像到指定目录。
    参数:
        original_dir (str): 原始图像目录，例如 'C:/.../data/all_slices'
        data_dir (str): 数据基础路径，例如 'C:/.../data'
    """
    # 定义目录路径
    voc2012_dir = os.path.join(data_dir, 'VOC2012')
    train_dir = os.path.join(voc2012_dir, 'train')
    val_dir = os.path.join(voc2012_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    srf_2_data_dir = os.path.join(test_dir, 'SRF_2', 'data')
    srf_2_target_dir = os.path.join(test_dir, 'SRF_2', 'target')
    srf_4_data_dir = os.path.join(test_dir, 'SRF_4', 'data')
    srf_4_target_dir = os.path.join(test_dir, 'SRF_4', 'target')
    srf_8_data_dir = os.path.join(test_dir, 'SRF_8', 'data')
    srf_8_target_dir = os.path.join(test_dir, 'SRF_8', 'target')

    # 创建所有需要的目录
    for dir_path in [train_dir, val_dir, srf_2_data_dir, srf_2_target_dir, 
                     srf_4_data_dir, srf_4_target_dir, srf_8_data_dir, srf_8_target_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 获取所有PNG文件并排序
    all_files = [f for f in os.listdir(original_dir) if is_image_file(f) and f.endswith('.png')]
    all_files.sort()  # 排序以确保一致性

    # 验证文件数量
    if len(all_files) < 22763:
        raise ValueError(f"图像数量不足，需要22763张，实际有{len(all_files)}张")
    
    # 分割为train, val, test
    train_files = all_files[:19380]
    val_files = all_files[19380:19380+1127]
    test_files = all_files[19380+1127:19380+1127+2256]

    # 处理train和val：仅保存HR图像
    for files, dest_dir in [(train_files, train_dir), (val_files, val_dir)]:
        for filename in files:
            image_path = os.path.join(original_dir, filename)
            image = Image.open(image_path).convert('L')  # 转换为灰度
            hr_image = resize_image(image, max_size=1024)  # 调整大小
            hr_image.save(os.path.join(dest_dir, filename))  # 保存HR图像

    # 处理test：生成不同分辨率的图像
    for filename in test_files:
        image_path = os.path.join(original_dir, filename)
        image = Image.open(image_path).convert('L')
        hr_image = resize_image(image, max_size=1024)  # 生成HR图像

        # 生成最小的LR图像（1/8 HR）
        lr_image_8 = generate_lr_image(hr_image, 8)
        if lr_image_8:
            # 保存到所有SRF的data目录
            for data_dir in [srf_2_data_dir, srf_4_data_dir, srf_8_data_dir]:
                lr_image_8.save(os.path.join(data_dir, filename))

        # 生成SRF_2的target：1/4 HR（从HR下采样4倍）
        target_image_2 = generate_lr_image(hr_image, 4)
        if target_image_2:
            target_image_2.save(os.path.join(srf_2_target_dir, filename))

        # 生成SRF_4的target：1/2 HR（从HR下采样2倍）
        target_image_4 = generate_lr_image(hr_image, 2)
        if target_image_4:
            target_image_4.save(os.path.join(srf_4_target_dir, filename))

        # 生成SRF_8的target：原始HR图像
        hr_image.save(os.path.join(srf_8_target_dir, filename))

if __name__ == "__main__":
    # 设置路径
    data_dir = r'C:\Users\13550\Downloads\SRGAN_changed\SRGAN_med_CT\data'
    original_dir = os.path.join(data_dir, 'all_slices')
    process_and_distribute_images(original_dir, data_dir)