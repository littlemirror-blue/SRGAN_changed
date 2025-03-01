import os
import shutil
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

def process_and_save_images(original_dir, base_path):
    """
    处理原始图像并生成不同上采样因子的低分辨率和高分辨率图像。
    参数:
        original_dir (str): 原始图像所在的目录路径
        base_path (str): 生成的目标基础路径，例如 'data/test'
    """
    # 定义目录路径
    srf_8_target_dir = os.path.join(base_path, 'SRF_8', 'target') 
    srf_8_data_dir = os.path.join(base_path, 'SRF_8', 'data')      
    srf_4_target_dir = os.path.join(base_path, 'SRF_4', 'target')  
    srf_2_target_dir = os.path.join(base_path, 'SRF_2', 'target')                                      
    
    # 创建所有需要的目录，包括 SRF_4/data 和 SRF_2/data
    for dir_path in [srf_8_target_dir, srf_8_data_dir, srf_4_target_dir, srf_2_target_dir,
                     os.path.join(base_path, 'SRF_4', 'data'), os.path.join(base_path, 'SRF_2', 'data')]:
        # 检查目录是否存在，若不存在则创建
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)  # 创建目录，支持多级目录创建
    
    # 遍历原始图像目录中的所有文件
    for filename in os.listdir(original_dir):
        # 检查文件是否为支持的图像格式
        if is_image_file(filename):
            # 构建原始图像的完整路径
            original_path = os.path.join(original_dir, filename)
            srf_8_target_path = os.path.join(srf_8_target_dir, filename)
            srf_8_data_path = os.path.join(srf_8_data_dir, filename)
            srf_4_target_path = os.path.join(srf_4_target_dir, filename)
            srf_2_target_path = os.path.join(srf_2_target_dir, filename)

            # 打开原始图像并转换为灰度模式（黑白通道）
            image = Image.open(original_path).convert('L')
            # 将图像分辨率调整到 1024x1024 以内，保持宽高比
            hr_image = resize_image(image, max_size=1024)
            # 保存调整后的高分辨率图像到 SRF_8/target
            hr_image.save(srf_8_target_path)

            # 生成上采样因子为 8 的低分辨率图像（min_size）
            lr_image_8 = generate_lr_image(hr_image, 8)
            if lr_image_8:  # 确保生成成功（避免分辨率过小导致失败）
                # 保存到 SRF_8/data
                lr_image_8.save(srf_8_data_path)

            # 生成上采样因子为 2 的图像（min_sizeX4），用于 SRF_4/target
            lr_image_4_target = generate_lr_image(hr_image, 2)
            if lr_image_4_target:  # 确保生成成功
                # 保存到 SRF_4/target
                lr_image_4_target.save(srf_4_target_path)

            # 生成上采样因子为 4 的图像（min_sizeX2），用于 SRF_2/target
            lr_image_2_target = generate_lr_image(hr_image, 4)
            if lr_image_2_target:  # 确保生成成功
                # 保存到 SRF_2/target
                lr_image_2_target.save(srf_2_target_path)

            # 将 SRF_8/data 的低分辨率图像复制到 SRF_4/data
            shutil.copy(srf_8_data_path, os.path.join(base_path, 'SRF_4', 'data', filename))
            # 将 SRF_8/data 的低分辨率图像复制到 SRF_2/data
            shutil.copy(srf_8_data_path, os.path.join(base_path, 'SRF_2', 'data', filename))
            # 打印复制成功的提示信息
            print(f"Copied SRF_8 data to SRF_4/data and SRF_2/data.")

if __name__ == "__main__":
    # 设置原始图像目录（需替换为实际路径）
    original_dir = 'dataset_original/original'
    # 设置目标基础路径
    base_path = 'data/test'
    # 调用处理函数
    process_and_save_images(original_dir, base_path)