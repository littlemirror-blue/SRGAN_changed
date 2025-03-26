import torch

def test_gpu():
    # 检查CUDA是否可用
    print("=" * 50)
    print("PyTorch GPU 测试")
    print("=" * 50)
    
    # 基本信息检查
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        # 设备数量和信息
        device_count = torch.cuda.device_count()
        print(f"可用的GPU数量: {device_count}")
        
        for i in range(device_count):
            print(f"\nGPU {i} 信息:")
            print(f"设备名称: {torch.cuda.get_device_name(i)}")
            print(f"计算能力: {torch.cuda.get_device_capability(i)}")
            print(f"总内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # 默认设备
        default_device = torch.cuda.current_device()
        print(f"\n默认使用的GPU: {default_device} ({torch.cuda.get_device_name(default_device)})")
        
        # 简单的计算测试
        print("\n运行GPU计算测试...")
        try:
            # 创建一个大的张量并在GPU上计算
            x = torch.randn(10000, 10000).cuda()
            y = torch.randn(10000, 10000).cuda()
            z = x @ y  # 矩阵乘法
            
            # 检查结果
            print("GPU计算测试成功完成!")
            print(f"结果张量形状: {z.shape}")
            print(f"结果张量的前5个值: {z[0, :5].cpu().numpy()}")
            
            # 内存使用情况
            print(f"\n当前GPU内存使用情况:")
            print(f"已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"已缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            
        except Exception as e:
            print(f"GPU计算测试失败: {str(e)}")
    else:
        print("\nCUDA不可用，PyTorch无法使用GPU")
        print("可能的原因:")
        print("1. 没有安装NVIDIA GPU驱动")
        print("2. 没有安装CUDA工具包")
        print("3. 安装的PyTorch版本不支持CUDA")
        print("4. 环境配置有问题")

if __name__ == "__main__":
    test_gpu()