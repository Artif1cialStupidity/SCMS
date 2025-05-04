# train_sc.py

import torch
import yaml
import argparse
import os
import pprint
import random
import sys

# --- 更新导入 ---
# 导入训练协调函数
from SC.train_victim import train_victim_model
# 导入配置加载工具 (假设 config_utils.py 在项目根目录下的 config/ 中)
try:
    from config.config_utils import load_config, save_config, pretty_print_config
except ImportError:
    print("错误：无法导入 config_utils。请确保 config/config_utils.py 文件存在。")
    # 定义一个临时的 load_config 以便脚本至少可以尝试运行（如果用户修复导入）
    def load_config(path):
        print(f"警告: 正在使用临时的 load_config 函数。请修复 config_utils 导入。")
        try:
            with open(path, 'r') as f: return yaml.safe_load(f)
        except Exception as e:
            print(f"加载配置 {path} 时出错: {e}")
            return None
    def save_config(cfg, path): pass # 临时占位
    def pretty_print_config(cfg): pprint.pprint(cfg) # 临时占位


# --- 参数解析 ---
def parse_args():
    """解析用于受害者模型训练的命令行参数。"""
    parser = argparse.ArgumentParser(description="训练语义通信 (受害者) 模型")
    parser.add_argument('--config', type=str, required=True,
                        help='受害者模型配置 YAML 文件的路径。')
    parser.add_argument('--output-dir', type=str, default='./results/victim_training_run',
                        help='保存训练结果（日志、最佳模型检查点、使用的配置）的目录。')
    # 可以添加其他覆盖配置的选项，例如 --epochs, --lr 等，但现在保持简单
    args = parser.parse_args()
    return args

# --- 主函数 ---
def main():
    args = parse_args()

    # --- 加载配置 ---
    config = load_config(args.config)
    if config is None:
        print(f"错误: 无法从 {args.config} 加载配置。退出。")
        sys.exit(1) # 配置文件加载失败是严重错误

    # --- 设置输出目录和保存路径 ---
    # 创建主输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    # 将模型检查点保存路径标准化到输出目录下的子文件夹中
    checkpoint_save_dir = os.path.join(args.output_dir, 'checkpoints')
    config['save_path'] = checkpoint_save_dir # 更新配置中的 save_path
    print(f"训练输出将保存到: {args.output_dir}")
    print(f"模型检查点将保存到: {checkpoint_save_dir}")

    # --- 打印和保存配置 ---
    print("\n--- 受害者训练配置 ---")
    pretty_print_config(config) # 使用导入的函数打印配置
    print("------------------------\n")

    # 保存本次运行使用的配置副本到输出目录
    # 使用输入配置文件的基本名来命名保存的文件，以便区分
    input_config_basename = os.path.basename(args.config)
    config_save_path = os.path.join(args.output_dir, f'used_{input_config_basename}')
    try:
        save_config(config, config_save_path) # 使用导入的函数保存
        print(f"使用的配置已保存到: {config_save_path}")
    except Exception as e:
        print(f"警告: 保存配置副本时出错: {e}")


    # --- 设置设备和随机种子 ---
    # (这部分逻辑也可以移到 train_victim_model 内部，但放在这里也 OK)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    seed = config.get('seed', 42) # 从配置中获取种子，默认 42
    print(f"使用随机种子: {seed}")
    torch.manual_seed(seed)
    random.seed(seed)
    if device == torch.device("cuda"):
        torch.cuda.manual_seed_all(seed)
        # 如果需要绝对可复现性（可能牺牲性能）
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # --- 开始训练 ---
    print("\n调用训练协调函数...")
    # train_victim_model 函数负责数据加载、模型初始化、训练循环、评估和保存最佳模型
    best_model_path = train_victim_model(config) # 传递加载并可能更新过的配置

    if best_model_path and os.path.exists(best_model_path):
        print(f"\n训练流程完成。")
        print(f"最佳受害者模型保存在: {best_model_path}")
        print(f"所有输出（日志、配置、检查点）位于: {args.output_dir}")
    else:
        print("\n训练流程完成，但无法确认最佳模型的保存路径或文件不存在。")
        print(f"请检查输出目录: {args.output_dir}")

if __name__ == '__main__':
     main()