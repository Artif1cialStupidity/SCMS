# train_sc.py

import torch
import yaml
import argparse
import os
import pprint
import random
import sys
import traceback # Import traceback

# --- 更新导入 ---
# 导入训练协调函数
from SC.train_victim import train_victim_model # <--- 确保导入的是修改后的函数
# 导入配置加载工具 (假设 config_utils.py 在项目根目录下的 config/ 中)
try:
    from config.config_utils import load_config, save_config, pretty_print_config
except ImportError:
    print("错误：无法导入 config_utils。请确保 config/config_utils.py 文件存在。")
    # Define temporary functions if import fails
    def load_config(path):
        print(f"警告: 正在使用临时的 load_config 函数。请修复 config_utils 导入。")
        try:
            with open(path, 'r') as f: return yaml.safe_load(f)
        except Exception as e:
            print(f"加载配置 {path} 时出错: {e}")
            return None
    def save_config(cfg, path):
        print(f"警告: 正在使用临时的 save_config 函数。无法保存配置到 {path}")
        pass
    def pretty_print_config(cfg): pprint.pprint(cfg)


# --- 参数解析 ---
def parse_args():
    """解析用于受害者模型训练的命令行参数。"""
    parser = argparse.ArgumentParser(description="训练语义通信 (受害者) 模型")
    parser.add_argument('--config', type=str, required=True,
                        help='受害者模型配置 YAML 文件的路径。')
    parser.add_argument('--output-dir', type=str, default='./results/victim_training_run',
                        help='保存训练结果（日志、最终检查点、使用的配置、历史记录、图表）的目录。') # <--- 更新帮助文本
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
    # !! 重要：更新配置字典中的 save_path，这样 train_victim_model 能获取到正确的检查点路径
    config['save_path'] = checkpoint_save_dir
    print(f"训练输出将保存到: {args.output_dir}")
    print(f"模型检查点将保存到: {checkpoint_save_dir}")

    # --- 打印和保存配置 ---
    print("\n--- 受害者训练配置 ---")
    pretty_print_config(config) # 使用导入的函数打印配置
    print("------------------------\n")

    # 保存本次运行使用的配置副本到输出目录
    input_config_basename = os.path.basename(args.config)
    config_save_path = os.path.join(args.output_dir, f'used_{input_config_basename}')
    try:
        save_config(config, config_save_path) # 使用导入的函数保存
        print(f"使用的配置已保存到: {config_save_path}")
    except Exception as e:
        print(f"警告: 保存配置副本时出错: {e}")


    # --- 设置设备和随机种子 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    seed = config.get('seed', 42) # 从配置中获取种子，默认 42
    print(f"使用随机种子: {seed}")
    torch.manual_seed(seed)
    random.seed(seed)
    if device == torch.device("cuda"):
        torch.cuda.manual_seed_all(seed)

    # --- 开始训练 ---
    print("\n调用训练协调函数...")
    # train_victim_model 函数负责数据加载、模型初始化、训练循环、评估和保存最佳模型
    # *** 将 args.output_dir 传递给 train_victim_model ***
    last_model_path = train_victim_model(config, args.output_dir) # <<< 变量名改为 last_model_path

    # --- 修改结束语 ---
    if last_model_path and os.path.exists(last_model_path):
        print(f"\n训练流程完成。")
        print(f"最后一个 epoch 的模型保存在: {last_model_path}") # <<< 更新文本
        # 最终评估结果已在 train_victim_model 中打印
        print(f"所有输出（日志、配置、最终检查点、历史记录、图表）位于: {args.output_dir}") # <<< 更新文本
    elif last_model_path: # Model path was returned but file doesn't exist
        print(f"\n训练流程完成，但无法在预期路径找到最后一个模型文件: {last_model_path}")
        print(f"请检查输出目录中的错误日志: {args.output_dir}")
    else: # train_victim_model likely returned None due to an error
        print("\n训练流程因错误未能成功完成。")
        print(f"请检查输出目录中的错误日志: {args.output_dir}")


if __name__ == '__main__':
     try:
         main()
     except Exception as e:
         print("\n--- Uncaught Exception in main ---")
         traceback.print_exc()
         print("------------------------------------\n")
         sys.exit(1)