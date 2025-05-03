# visualize_sc_output.py

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import yaml
import argparse
import os
import numpy as np

# 假设这些模块在你的项目结构中可以被导入
from SC.res_model import SC_Model
from SC.base_model import Base_Model
from data.cifar import get_cifar_loaders, CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD # 导入均值和标准差
from config.config_utils import load_config

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Visualize SC model input (x) and output (x')")
    parser.add_argument('--victim-checkpoint', type=str, required=True,
                        help='Path to the pre-trained victim SC model checkpoint (.pth file).')
    parser.add_argument('--victim-config', type=str, required=True,
                        help='Path to the original configuration YAML file used to train the victim model.')
    parser.add_argument('--num-images', type=int, default=5,
                        help='Number of images to display.')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Optional: Path to save the visualization image.')
    args = parser.parse_args()
    return args

def denormalize(tensor, mean, std):
    """将归一化的图像张量反归一化回 [0, 1] 范围以便显示"""
    # 防止原地修改原始张量
    tensor = tensor.clone().cpu()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m) # 反归一化: t = t * std + mean
    # 将张量维度从 (C, H, W) 转换为 (H, W, C) 以便 matplotlib 显示
    tensor = tensor.permute(1, 2, 0)
    # 将数值范围裁剪到 [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.numpy() # 转换为 NumPy 数组

def main():
    args = parse_args()

    # --- 加载配置 ---
    victim_config = load_config(args.victim_config)
    if victim_config is None:
        print(f"错误：无法加载受害者配置文件 {args.victim_config}")
        return
    if victim_config['task'] != 'reconstruction':
        print(f"警告：此脚本主要设计用于重建任务的可视化。")
        # 对于分类任务，输出 y 不是图像，但仍可尝试运行

    # --- 设置设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 加载数据 (获取测试集) ---
    dataset_name = victim_config['dataset']['name']
    data_dir = victim_config['dataset'].get('data_dir', f'./data_{dataset_name}')
    try:
        # 只需要测试加载器来获取样本
        # 使用稍大的 batch_size 确保能获取足够图像，但只取需要的数量
        _, test_loader = get_cifar_loaders(
            dataset_name=dataset_name,
            batch_size=max(args.num_images, 10), # 获取一个稍大的批次
            data_dir=data_dir,
            augment_train=False # 测试集不增强
        )
    except ValueError as e:
        print(f"加载数据集时出错: {e}")
        return

    # --- 加载模型 ---
    print(f"加载受害者模型结构，使用配置: {args.victim_config}")
    print(f"加载受害者模型权重: {args.victim_checkpoint}")

    # 根据配置初始化模型结构
    num_classes = 10 if dataset_name == 'cifar10' else 100
    decoder_conf_victim = victim_config['victim_model']['decoder'].copy()
    if victim_config['task'] == 'classification':
        if 'num_classes' not in decoder_conf_victim:
            decoder_conf_victim['num_classes'] = num_classes
            print(f"Automatically set num_classes to {num_classes} in decoder config")


    model_type = victim_config['victim_model'].get('type', 'resnet_sc').lower()
    print(f"Attempting to initialize victim model of type: '{model_type}' for visualization")

    try:
        if model_type == 'base_model':
            print("Initializing Base_Model...")
            model = Base_Model(
                encoder_config=victim_config['victim_model']['encoder'],
                channel_config=victim_config['channel'],
                decoder_config=decoder_conf_victim,
                task=victim_config['task']
            ).to(device)
            print("Base_Model initialized successfully.")

        elif model_type == 'resnet_sc':
            print("Initializing SC_Model (ResNet based)...")
            model = SC_Model(
                encoder_config=victim_config['victim_model']['encoder'],
                channel_config=victim_config['channel'],
                decoder_config=decoder_conf_victim,
                task=victim_config['task']
            ).to(device)
            print("SC_Model (ResNet) initialized successfully.")

        else:
            raise ValueError(f"Unknown victim model type specified in config: '{model_type}'. Choose 'base_model' or 'resnet_sc'.")

        # 加载权重 (移到 try 块内部，因为必须在模型实例化后加载)
        model.load_state_dict(torch.load(args.victim_checkpoint, map_location=device))
        model.eval() # ****** 设置为评估模式 ******
        print("受害者模型权重加载成功。")

    except FileNotFoundError:
        print(f"错误: 受害者模型权重文件未找到 {args.victim_checkpoint}")
        return
    except ValueError as ve: # 捕获我们自己抛出的 ValueError
        print(f"配置错误: {ve}")
        return
    except Exception as e:
        print(f"加载受害者模型时发生错误: {e}")
        # 打印更详细的错误跟踪信息
        import traceback
        traceback.print_exc()
        return

    # --- 获取样本数据并进行推理 ---
    print(f"从测试集获取 {args.num_images} 张图像并进行推理...")
    try:
        data_batch, _ = next(iter(test_loader)) # 获取一个批次的数据和标签（标签在这里不用）
        sample_x = data_batch[:args.num_images].to(device) # 取需要的数量并移到设备
    except StopIteration:
        print("错误：无法从数据加载器获取数据。")
        return

    with torch.no_grad(): # 推理时不需要计算梯度
        # 通过模型获取重建图像 x' (对于重建任务，模型输出即 x')
        sample_x_prime = model(sample_x)

    # 将数据移回 CPU 以便可视化
    sample_x = sample_x.cpu()
    sample_x_prime = sample_x_prime.cpu()

    # --- 可视化 ---
    print("生成可视化图像...")
    # 选择正确的均值和标准差进行反归一化
    if dataset_name == 'cifar10':
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif dataset_name == 'cifar100':
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    else:
        print(f"警告：未知的 dla {dataset_name} 的均值/标准差，使用 CIFAR-10 的值。")
        mean, std = CIFAR10_MEAN, CIFAR10_STD

    fig, axs = plt.subplots(args.num_images, 2, figsize=(6, 2 * args.num_images)) # 每行显示一对图

    for i in range(args.num_images):
        # 反归一化原始图像和重建图像
        original_img = denormalize(sample_x[i], mean, std)
        reconstructed_img = denormalize(sample_x_prime[i], mean, std)

        # 显示原始图像
        ax = axs[i, 0] if args.num_images > 1 else axs[0]
        ax.imshow(original_img)
        ax.set_title(f"Original (x) #{i+1}")
        ax.axis('off') # 关闭坐标轴

        # 显示重建图像
        ax = axs[i, 1] if args.num_images > 1 else axs[1]
        ax.imshow(reconstructed_img)
        ax.set_title(f"Reconstructed (x') #{i+1}")
        ax.axis('off') # 关闭坐标轴

    plt.tight_layout() # 调整子图布局

    # 保存或显示图像
    if args.output_file:
        try:
            plt.savefig(args.output_file)
            print(f"可视化结果已保存到: {args.output_file}")
        except Exception as e:
            print(f"保存图像时出错: {e}")
    else:
        plt.show()

if __name__ == '__main__':
    main()