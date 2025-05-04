# visualize_sc_output.py

import torch
import torchvision
import matplotlib.pyplot as plt
import yaml
import argparse
import os
import numpy as np
import sys

# --- 更新导入 ---
# 从新的 models 目录导入统一的 SC_Model
try:
    from models.model import SC_Model
except ImportError:
    print("错误: 无法从 'models.sc_model' 导入 SC_Model。请确保路径正确。")
    sys.exit(1)
# 数据加载工具和常量 (导入均值/标准差用于反归一化)
try:
    from data.cifar import get_cifar_loaders, CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD
except ImportError:
     print("错误: 无法从 'data.cifar' 导入。请确保路径正确。")
     sys.exit(1)
# 配置加载工具
try:
    from config.config_utils import load_config
except ImportError:
    print("警告: 无法从 'config.config_utils' 导入 load_config。使用临时函数。")
    def load_config(path):
        try:
            with open(path, 'r') as f: return yaml.safe_load(f)
        except Exception as e: print(f"加载配置 {path} 时出错: {e}"); return None


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="可视化 SC 模型输入 (x) 和输出 (x')")
    parser.add_argument('--victim-checkpoint', type=str, required=True,
                        help='预训练受害者 SC 模型检查点 (.pth 文件) 的路径。')
    parser.add_argument('--victim-config', type=str, required=True,
                        help='用于训练该受害者模型的原始配置 YAML 文件的路径。')
    parser.add_argument('--num-images', type=int, default=5,
                        help='要显示的图像数量。')
    parser.add_argument('--output-file', type=str, default=None,
                        help='可选：保存可视化图像的文件路径。')
    args = parser.parse_args()
    return args

def denormalize(tensor, mean, std):
    """将归一化的图像张量反归一化回 [0, 1] 范围以便显示。"""
    # 防止原地修改原始张量
    tensor = tensor.clone().cpu() # 确保在 CPU 上操作
    # 确保 mean 和 std 是 tensor 形式，并且维度匹配
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)

    # 反归一化: t = t * std + mean
    tensor.mul_(std).add_(mean)

    # 将张量维度从 (C, H, W) 转换为 (H, W, C) 以便 matplotlib 显示
    tensor = tensor.permute(1, 2, 0)
    # 将数值范围裁剪到 [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.numpy() # 转换为 NumPy 数组

def main():
    args = parse_args()

    # --- 加载受害者配置 ---
    print(f"加载受害者配置: {args.victim_config}")
    victim_config = load_config(args.victim_config)
    if victim_config is None:
        print(f"错误：无法加载受害者配置文件。退出。")
        sys.exit(1)

    task = victim_config.get('task', 'unknown').lower()
    if task != 'reconstruction':
        print(f"警告：此脚本主要设计用于'重建(reconstruction)'任务的可视化。")
        print(f"当前任务是 '{task}'，将尝试显示模型的输出，但这可能不是图像。")

    # --- 设置设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 加载数据 (获取测试集样本) ---
    dataset_name = victim_config['dataset']['name']
    data_dir = victim_config['dataset'].get('data_dir', f'./data_{dataset_name}')
    print(f"加载数据集: {dataset_name}")
    try:
        # 使用一个足够大的批次来获取所需数量的图像，但只使用 num_images 个
        # 注意：get_cifar_loaders 返回 train_loader, test_loader
        _, test_loader = get_cifar_loaders(
            dataset_name=dataset_name,
            batch_size=max(args.num_images, 32), # 获取一个合理的批次
            data_dir=data_dir,
            augment_train=False, # 测试集不增强
            num_workers=0 # 可视化时通常不需要多进程
        )
    except ValueError as e:
        print(f"加载数据集时出错: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"加载数据集时发生意外错误: {e}")
        sys.exit(1)

    # --- 加载模型 ---
    print(f"初始化统一 SC 模型...")
    try:
        # 使用统一的 SC_Model 和完整的配置进行实例化
        model = SC_Model(config=victim_config).to(device)
        print("模型结构初始化成功。")
        # 加载预训练权重
        print(f"加载模型权重: {args.victim_checkpoint}")
        model.load_state_dict(torch.load(args.victim_checkpoint, map_location=device))
        model.eval() # ****** 非常重要：设置为评估模式 ******
        print("受害者模型权重加载成功。")

    except FileNotFoundError:
        print(f"错误: 受害者模型权重文件未找到 {args.victim_checkpoint}")
        sys.exit(1)
    except (KeyError, ValueError) as e: # 捕获模型初始化或配置错误
        print(f"模型初始化或配置错误: {e}")
        print("请确保配置文件和权重文件匹配，且配置文件包含所有必需项（如 'victim_model.type'）。")
        sys.exit(1)
    except Exception as e:
        print(f"加载受害者模型时发生意外错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- 获取样本数据并进行推理 ---
    print(f"从测试集获取 {args.num_images} 张图像并进行推理...")
    try:
        data_iter = iter(test_loader)
        data_batch, _ = next(data_iter) # 获取一个批次的数据和标签（标签在这里不用）
        if data_batch.size(0) < args.num_images:
            print(f"警告：测试集批次大小 ({data_batch.size(0)}) 小于请求的图像数量 ({args.num_images})。将只显示 {data_batch.size(0)} 张。")
            args.num_images = data_batch.size(0)
        sample_x = data_batch[:args.num_images].to(device) # 取需要的数量并移到设备
    except StopIteration:
        print("错误：无法从数据加载器获取数据。测试集为空或批次大小为 0？")
        sys.exit(1)
    except Exception as e:
        print(f"获取数据时出错: {e}")
        sys.exit(1)


    print("通过模型进行前向传播...")
    with torch.no_grad(): # 推理时不需要计算梯度
        # 通过模型获取输出 y (对于重建任务，y 就是 x')
        output_y = model(sample_x) # 使用统一模型接口

    # 将数据移回 CPU 以便可视化
    sample_x_cpu = sample_x.cpu()
    output_y_cpu = output_y.cpu()

    # --- 可视化 ---
    print("生成可视化图像...")
    # 选择正确的均值和标准差进行反归一化
    if dataset_name == 'cifar10':
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif dataset_name == 'cifar100':
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    else:
        print(f"警告：未知的 {dataset_name} 的均值/标准差，使用 CIFAR-10 的默认值。")
        mean, std = CIFAR10_MEAN, CIFAR10_STD

    # 创建绘图区域
    # 行数等于图像数，列数为 2 (原始 vs 输出)
    fig, axs = plt.subplots(args.num_images, 2, figsize=(6, 2.5 * args.num_images))
    # 如果只有一张图片，axs 不是数组，需要特殊处理
    if args.num_images == 1:
        axs = np.array([axs]) # 转换成 NumPy 数组以便索引

    fig.suptitle(f'{dataset_name.upper()} - Original vs Model Output ({task.capitalize()})', fontsize=14)

    for i in range(args.num_images):
        # 反归一化原始图像
        original_img = denormalize(sample_x_cpu[i], mean, std)

        # 处理模型输出
        output_display = None
        title_output = f"Output (Y) #{i+1}"
        if task == 'reconstruction':
            # 对于重建任务，输出是图像，也需要反归一化
            try:
                output_display = denormalize(output_y_cpu[i], mean, std)
                title_output = f"Reconstructed (X') #{i+1}"
            except Exception as e:
                 print(f"警告: 反归一化重建图像时出错 for image {i}: {e}. 将尝试直接显示。")
                 output_display = output_y_cpu[i].permute(1, 2, 0).numpy() # 尝试直接转换格式

        else:
            # 对于非重建任务 (如分类)，输出不是图像
            # 可以在这里显示其他信息，或者留空/显示提示
            output_display = np.ones_like(original_img) * 0.8 # 显示灰色背景
            # 可以尝试在图上显示分类结果或 logits
            if task == 'classification' and output_y_cpu.ndim == 2: # 检查输出是否是 logits (B, num_classes)
                 pred_class = torch.argmax(output_y_cpu[i]).item()
                 title_output = f"Pred Class: {pred_class}"
                 # (可以添加代码将 logits 文本添加到灰色图像上)


        # 显示原始图像
        ax_orig = axs[i, 0]
        ax_orig.imshow(original_img)
        ax_orig.set_title(f"Original (X) #{i+1}")
        ax_orig.axis('off') # 关闭坐标轴

        # 显示模型输出
        ax_out = axs[i, 1]
        if output_display is not None:
             # 检查图像是否为单通道灰度图，如果是，设置 colormap
             if output_display.ndim == 2 or (output_display.ndim == 3 and output_display.shape[2] == 1):
                 ax_out.imshow(output_display.squeeze(), cmap='gray')
             else:
                 ax_out.imshow(output_display)
        ax_out.set_title(title_output)
        ax_out.axis('off') # 关闭坐标轴

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # 调整子图布局，留出顶部空间给 suptitle

    # 保存或显示图像
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir: # 确保目录存在
             os.makedirs(output_dir, exist_ok=True)
        try:
            plt.savefig(args.output_file, dpi=150) # 提高分辨率
            print(f"可视化结果已保存到: {args.output_file}")
        except Exception as e:
            print(f"保存图像时出错: {e}")
    else:
        print("显示可视化结果...")
        plt.show()

if __name__ == '__main__':
    main()