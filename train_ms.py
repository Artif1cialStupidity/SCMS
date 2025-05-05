# train_ms.py (用于运行模型窃取攻击)

import torch
import yaml
import argparse
import os
import pprint
import random
import json
import sys
import traceback

# --- 统一导入块 ---
try:
    # 数据加载
    from data.cifar import get_cifar_loaders
    # 受害者模型定义 (统一模型)
    from models.model import SC_Model
    # 攻击组件
    from MS.query_interface import VictimQueryInterface
    from MS.attacker import ModelStealingAttacker
    # 配置工具
    from config.config_utils import load_config, save_config, pretty_print_config
    # 替代编码器定义 (如果需要在脚本中明确类型，但通常 Attacker 内部处理)
    # from models.components.resnet_components import ResNetEncoderSC
except ImportError as e:
     # 提供更详细的错误信息
     print(f"导入模块时出错: {e}")
     print("请确认以下模块/包存在于您的 Python 环境和路径中:")
     print("- data.cifar")
     print("- models.model")
     print("- MS.query_interface, MS.attacker")
     print("- config.config_utils")
     # 如果需要，取消注释下面一行以查看详细的回溯信息
     # traceback.print_exc()
     sys.exit(1) # 导入失败是致命错误，退出

# --- 参数解析 ---
def parse_args():
    """解析用于运行模型窃取攻击的命令行参数。"""
    parser = argparse.ArgumentParser(description="运行针对 SC 模型编码器的模型窃取攻击")
    parser.add_argument('--attack-config', type=str, required=True,
                        help='攻击配置 YAML 文件的路径 (必须包含 attacker 部分)。')
    parser.add_argument('--victim-checkpoint', type=str, required=True,
                        help='预训练受害者 SC 模型检查点 (.pth 文件) 的路径。')
    parser.add_argument('--victim-config', type=str, required=True,
                        help='用于训练该受害者模型的原始配置 YAML 文件的路径。')
    parser.add_argument('--output-dir', type=str, default='./results/ms_attack_run',
                        help='保存攻击结果 (替代编码器模型, 日志, 指标等) 的目录。')
    args = parser.parse_args()
    return args

# --- 主函数 ---
def main():
    args = parse_args()

    # --- 1. 加载配置 ---
    print("--- 加载配置 ---")
    attack_config_full = load_config(args.attack_config)
    victim_config = load_config(args.victim_config)
    if attack_config_full is None or victim_config is None:
        print("错误: 无法加载一个或多个配置文件。请检查路径。退出。")
        sys.exit(1)

    # 提取 attacker 子配置，并检查是否存在
    if 'attacker' not in attack_config_full:
         print(f"错误: 攻击配置文件 '{args.attack_config}' 必须包含 'attacker' 根键。")
         sys.exit(1)
    attack_config = attack_config_full['attacker'] # 使用 attacker 子字典

    # --- 2. 验证攻击配置 (针对编码器窃取) ---
    print("--- 验证攻击配置 ---")
    attack_type = attack_config.get('type')
    latent_access = attack_config.get('latent_access')
    query_access = attack_config.get('query_access')
    noise_scale_provided = 'noise_scale' in attack_config

    valid_config = True
    if attack_type != 'steal_encoder':
        print(f"错误: 当前实现仅支持 'type: steal_encoder'。找到: '{attack_type}'")
        valid_config = False
    if query_access not in ['encoder_query', 'end_to_end_query']:
         print(f"错误: 对于编码器窃取，'query_access' 必须是 'encoder_query' 或 'end_to_end_query'。找到: '{query_access}'")
         valid_config = False
    if latent_access == 'noisy_scaled_z':
        if not noise_scale_provided:
            print("错误: 当 'latent_access' 为 'noisy_scaled_z' 时，必须在配置中提供 'noise_scale'。")
            valid_config = False
    elif latent_access == 'clean_z':
        print("警告: 使用 'latent_access: clean_z'。'noise_scale' (如果提供) 将被忽略。这是一个基线比较场景。")
        # 如果 clean_z 时未提供 noise_scale，为其添加一个默认值以简化后续代码
        if not noise_scale_provided: attack_config['noise_scale'] = 0.0
    else:
        print(f"错误: 对于此设置，'latent_access' 必须是 'noisy_scaled_z' 或 'clean_z'。找到: '{latent_access}'")
        valid_config = False

    if not valid_config:
        print("配置验证失败。退出。")
        sys.exit(1)
    print("攻击配置验证通过。")

    # --- 3. 设置输出目录和保存配置 ---
    print(f"--- 设置输出目录: {args.output_dir} ---")
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as e:
        print(f"错误: 创建输出目录 '{args.output_dir}' 失败: {e}")
        sys.exit(1)

    print("保存使用的配置文件...")
    save_config(attack_config_full, os.path.join(args.output_dir, 'attack_config_used.yaml'))
    save_config(victim_config, os.path.join(args.output_dir, 'victim_config_used.yaml'))

    print("\n--- 攻击配置 (Attacker Block) 预览 ---")
    pretty_print_config(attack_config)
    print("--------------------------------------\n")

    # --- 4. 设置设备和随机种子 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = attack_config_full.get('seed', random.randint(0, 10000)) # 从顶层获取种子，若无则随机
    print(f"--- 设备: {device} | 随机种子: {seed} ---")
    torch.manual_seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # --- 5. 加载数据集 ---
    print("--- 加载数据集 ---")
    # 5.1 测试加载器 (基于受害者配置，用于最终评估)
    victim_dataset_name = victim_config.get('dataset', {}).get('name')
    victim_data_dir = victim_config.get('dataset', {}).get('data_dir')
    if not victim_dataset_name or not victim_data_dir:
        print("错误: 受害者配置中缺少 'dataset.name' 或 'dataset.data_dir'。")
        sys.exit(1)
    eval_batch_size = attack_config.get('training_attacker', {}).get('batch_size', 128)
    print(f"加载测试数据集: {victim_dataset_name} (用于评估)")
    try:
        _, test_loader = get_cifar_loaders(
            dataset_name=victim_dataset_name,
            batch_size=eval_batch_size,
            data_dir=victim_data_dir,
            augment_train=False # 测试时不增强
        )
    except Exception as e:
        print(f"错误: 加载测试数据集 '{victim_dataset_name}' 失败: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5.2 代理查询加载器 (优先使用攻击配置，否则回退到受害者配置)
    proxy_loader_batch_size = attack_config.get('training_attacker', {}).get('query_batch_size', 128)
    proxy_config = attack_config.get('proxy_dataset')
    if proxy_config and 'name' in proxy_config and 'data_dir' in proxy_config:
        proxy_dataset_name = proxy_config['name']
        proxy_data_dir = proxy_config['data_dir']
        print(f"加载代理数据集 (来自 attack_config): {proxy_dataset_name} from {proxy_data_dir}")
    else:
        proxy_dataset_name = victim_dataset_name
        proxy_data_dir = victim_data_dir
        print(f"警告: 未在 attack_config 中指定 'proxy_dataset'。回退使用受害者数据集作为代理: {proxy_dataset_name}")

    try:
        # 假设代理数据集也用 get_cifar_loaders 加载。如果不同需要修改此处逻辑。
        # 代理查询通常也不需要数据增强
        _, proxy_loader = get_cifar_loaders(
             dataset_name=proxy_dataset_name,
             batch_size=proxy_loader_batch_size,
             data_dir=proxy_data_dir,
             augment_train=False
         )
        print(f"代理查询加载器 ({proxy_dataset_name}) 创建成功。")
    except Exception as e:
        print(f"错误: 创建代理查询加载器 ({proxy_dataset_name}) 失败: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- 6. 加载预训练的受害者模型 ---
    print(f"\n--- 加载受害者模型 ---")
    print(f"使用配置: {args.victim_config}")
    print(f"加载权重: {args.victim_checkpoint}")
    try:
        # 使用统一模型类和完整的受害者配置实例化
        victim_model = SC_Model(config=victim_config).to(device)
    except (KeyError, ValueError) as e:
        print(f"错误: 从配置初始化受害者 SC_Model 结构失败: {e}")
        print("请确保受害者配置文件有效且包含所有必需字段 (例如 'victim_model.type')。")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"错误: 初始化受害者 SC_Model 时发生意外错误: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 加载权重
    if not os.path.exists(args.victim_checkpoint):
        print(f"错误: 受害者模型权重文件未找到: {args.victim_checkpoint}")
        sys.exit(1)
    try:
        victim_model.load_state_dict(torch.load(args.victim_checkpoint, map_location=device))
        victim_model.eval() # 关键：设置为评估模式
        print("受害者模型加载成功。")
    except Exception as e:
         print(f"错误: 加载受害者模型权重失败: {e}")
         traceback.print_exc()
         sys.exit(1)

    # --- 7. 初始化攻击者组件 ---
    print("\n--- 初始化攻击者 (编码器窃取模式) ---")
    try:
        # 7.1 查询接口
        victim_interface = VictimQueryInterface(victim_model, attack_config)

        # 7.2 攻击执行器 (Attacker)
        # 提取替代模型配置和训练配置
        surrogate_conf = attack_config.get('surrogate_model')
        attacker_train_conf = attack_config.get('training_attacker')
        if not surrogate_conf or not attacker_train_conf:
            print("错误: 攻击配置中缺少 'surrogate_model' 或 'training_attacker' 部分。")
            sys.exit(1)
        if 'latent_dim' not in surrogate_conf:
             print("错误: 'surrogate_model' 配置中必须包含 'latent_dim'。")
             sys.exit(1)

        attacker = ModelStealingAttacker(
            attack_config=attack_config,
            victim_interface=victim_interface,
            surrogate_model_config=surrogate_conf,
            attacker_train_config=attacker_train_conf,
            proxy_dataloader=proxy_loader,
            device=device
        )
        print("攻击者初始化成功。")
    except (ValueError, KeyError) as e:
         print(f"错误: 初始化攻击者组件失败: {e}")
         traceback.print_exc()
         sys.exit(1)
    except Exception as e:
         print(f"错误: 初始化攻击者时发生意外错误: {e}")
         traceback.print_exc()
         sys.exit(1)

    # --- 8. 运行攻击 ---
    print("\n--- 开始执行模型窃取攻击 ---")
    try:
        attacker.run_attack() # 执行数据收集和替代模型训练
    except Exception as e:
        print("\n!!! 攻击执行过程中发生错误 !!!")
        traceback.print_exc()
        print("尝试继续执行后续步骤 (保存和评估可能失败)...")
        # 根据需要决定是否在这里退出 sys.exit(1)

    # --- 9. 保存替代编码器模型 ---
    surrogate_save_path = None # 初始化路径
    if hasattr(attacker, 'surrogate_encoder') and attacker.surrogate_encoder is not None:
        surrogate_save_path = os.path.join(args.output_dir, 'surrogate_encoder.pth')
        print(f"\n--- 保存替代编码器模型至: {surrogate_save_path} ---")
        try:
            torch.save(attacker.surrogate_encoder.state_dict(), surrogate_save_path)
            print("替代编码器保存成功。")
        except Exception as e:
            print(f"错误: 保存替代编码器模型失败: {e}")
            surrogate_save_path = "保存失败" # 更新状态
    else:
        print("错误: 在攻击者对象上找不到 'surrogate_encoder' 或其为 None。无法保存模型。")
        surrogate_save_path = "未找到模型"

    # --- 10. 评估攻击效果 ---
    print("\n--- 评估攻击效果 (编码器保真度) ---")
    attack_eval_results = {} # 初始化结果字典
    try:
        # 确保测试加载器可用
        if 'test_loader' in locals():
             attack_eval_results = attacker.evaluate_attack(test_loader)
             print("评估完成。")
        else:
             print("错误: 'test_loader' 未定义，无法进行评估。")
             attack_eval_results = {"error": "test_loader not available"}
    except Exception as e:
        print(f"错误: 评估攻击效果时发生错误: {e}")
        traceback.print_exc()
        attack_eval_results = {"error": f"评估失败: {e}"}

    # --- 11. 保存最终结果 ---
    results_save_path = os.path.join(args.output_dir, 'attack_results.json')
    print(f"\n--- 保存最终攻击结果至: {results_save_path} ---")

    # 序列化结果 (将 Tensor 转为 float)
    serializable_results = {}
    for key, value in attack_eval_results.items():
         if isinstance(value, torch.Tensor):
             serializable_results[key] = value.item()
         elif isinstance(value, (int, float, str, bool)) or value is None:
              serializable_results[key] = value
         elif hasattr(value, 'item'): # 处理 numpy scalar 等
              try: serializable_results[key] = value.item()
              except: serializable_results[key] = str(value) # 最后手段转字符串
         else:
              serializable_results[key] = str(value) # 其他类型转字符串

    # 构建最终 JSON 对象
    final_results_summary = {
        'attack_config_summary': {
            'type': attack_config.get('type', 'N/A'),
            'query_access': attack_config.get('query_access', 'N/A'),
            'latent_access': attack_config.get('latent_access', 'N/A'),
            'noise_scale': attack_config.get('noise_scale', 'N/A'),
            'query_budget': attack_config.get('query_budget', 'N/A'),
            'final_query_count': victim_interface.get_query_count() if 'victim_interface' in locals() else 'N/A',
            'proxy_dataset_used': proxy_dataset_name if 'proxy_dataset_name' in locals() else 'N/A',
            'surrogate_encoder_arch': surrogate_conf.get('arch_name', 'N/A') if 'surrogate_conf' in locals() else 'N/A'
        },
        'evaluation_metrics': serializable_results,
        'surrogate_model_saved_path': surrogate_save_path if surrogate_save_path else "保存失败或未找到"
    }

    # 写入 JSON 文件
    try:
        with open(results_save_path, 'w') as f:
            json.dump(final_results_summary, f, indent=4)
        print("攻击结果 JSON 文件保存成功。")
    except Exception as e:
        print(f"错误: 保存攻击结果 JSON 文件失败: {e}")

    print("\n--- 模型窃取攻击流程结束 ---")
    print(f"所有输出保存在: {args.output_dir}")

# --- 脚本入口点 ---
if __name__ == '__main__':
     main()