# SC/train_victim.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import random
from tqdm import tqdm # 用于进度条
import json # <--- 添加 json 用于保存历史记录
import matplotlib.pyplot as plt # <--- 添加 matplotlib 用于绘图
import traceback # For more detailed error printing

# --- 更新导入 ---
# 从新的 models 目录导入统一的 SC_Model
from models.model import SC_Model
# 数据加载和评估指标的导入保持不变
from data.cifar import get_cifar_loaders
from evaluation.metrics import calculate_psnr, calculate_accuracy

# --- 辅助函数获取损失函数 ---
def get_victim_loss_criterion(task: str, loss_type: str = 'default'):
    """获取受害者模型的损失函数。"""
    if task == 'reconstruction':
        if loss_type.lower() == 'mse' or loss_type.lower() == 'default':
            print("使用 MSE Loss 进行重建。")
            return nn.MSELoss()
        elif loss_type.lower() == 'l1':
            print("使用 L1 Loss 进行重建。")
            return nn.L1Loss()
        else:
            raise ValueError(f"不支持的重建损失类型: {loss_type}")
    elif task == 'classification':
        print("使用 CrossEntropy Loss 进行分类。")
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"不支持的任务以获取损失标准: {task}")

# --- 训练循环 ---
def train_victim_epoch(model: SC_Model,
                       loader: DataLoader,
                       optimizer: optim.Optimizer,
                       criterion: nn.Module,
                       device: torch.device,
                       task: str):
    """为受害者 SC 模型运行一个训练周期。"""
    model.train() # 设置模型为训练模式
    total_loss = 0.0
    num_samples = 0

    progress_bar = tqdm(loader, desc=f'训练 Epoch', leave=False)
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)
        num_samples += batch_size

        optimizer.zero_grad()

        # 前向传播 (统一模型接口)
        output = model(data) # 不需要返回潜变量进行基础训练

        # 根据任务计算损失
        if task == 'reconstruction':
            loss = criterion(output, data) # 重建任务比较输出和原始输入
        elif task == 'classification':
            loss = criterion(output, target) # 分类任务比较 logits 和目标类别
        else:
             raise ValueError(f"在损失计算过程中遇到未知任务 '{task}'")

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size # 累加加权损失

        # 更新进度条
        progress_bar.set_postfix(loss=f'{loss.item():.4f}')

    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    return avg_loss

# --- 评估循环 (现在只在最后调用) ---
def evaluate_victim(model: SC_Model,
                    loader: DataLoader,
                    criterion: nn.Module,
                    device: torch.device,
                    task: str) -> dict:
    """在数据集上评估受害者 SC 模型。"""
    model.eval() # 设置模型为评估模式
    total_loss = 0.0
    num_samples = 0
    correct_predictions = 0
    total_psnr = 0.0

    results = {}

    progress_bar = tqdm(loader, desc=f'Final Evaluation', leave=False)
    with torch.no_grad(): # 评估时禁用梯度计算
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            num_samples += batch_size

            # 前向传播 (统一模型接口)
            output = model(data)

            # 计算损失
            if task == 'reconstruction':
                loss = criterion(output, data)
                total_loss += loss.item() * batch_size
                # --- 移除调试打印 ---
                # print("psnr",torch.Tensor)
                psnr_batch = calculate_psnr(output, data, data_range=2.0) # 假设数据范围是 [-1, 1]
                # Handle potential inf PSNR safely for averaging
                if psnr_batch != float('inf'):
                    total_psnr += psnr_batch * batch_size
                else:
                    # If perfect reconstruction, add a large value contribution for averaging
                     total_psnr += 100.0 * batch_size
                # --- 移除 set_postfix 或使其可选 ---
                # progress_bar.set_postfix(loss=f'{loss.item():.4f}', psnr=f'{psnr_batch:.2f}')

            elif task == 'classification':
                loss = criterion(output, target)
                total_loss += loss.item() * batch_size
                acc_batch, correct_batch = calculate_accuracy(output, target)
                correct_predictions += correct_batch
                # --- 移除 set_postfix 或使其可选 ---
                # progress_bar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{acc_batch*100:.2f}%')

    # 计算平均指标
    results['loss'] = total_loss / num_samples if num_samples > 0 else 0.0
    if task == 'reconstruction':
        avg_psnr = total_psnr / num_samples if num_samples > 0 else 0.0
        results['psnr'] = avg_psnr
    elif task == 'classification':
        results['accuracy'] = correct_predictions / num_samples if num_samples > 0 else 0.0

    return results


# --- 改动：绘图函数现在只绘制训练 Loss ---
def plot_training_history(history: dict, output_dir: str):
    """根据训练历史数据生成并保存训练 Loss 曲线图。"""
    epochs = history.get('epochs', [])
    train_losses = history.get('train_loss', [])

    if not epochs or not train_losses:
        print("No training history data to plot.")
        return

    # Use a style less likely to cause issues, like 'ggplot' or 'default'
    try:
        # Use a style guaranteed to exist
        plt.style.use('default')
        # Or try a common one, handle exception if not found
        # plt.style.use('ggplot')
    except OSError:
         print("Warning: Selected style not found, using default Matplotlib style.")
         plt.style.use('default') # Fallback to default


    # --- 绘制 Loss 曲线 ---
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', linestyle='-')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(output_dir, 'training_loss_curve.png') # Renamed file
    try:
        plt.savefig(loss_plot_path)
        print(f"Training loss curve saved to: {loss_plot_path}")
    except Exception as e:
        print(f"Error saving loss curve plot: {e}")
    plt.close() # 关闭图形，释放内存


# --- 主要训练协调函数 (重大改动) ---
def train_victim_model(config: dict, output_dir: str):
    """
    协调 SC 受害者模型训练的主函数。
    - 不进行每轮评估。
    - 保存最后一个 epoch 的模型。
    - 在训练结束后进行一次最终评估。
    """
    print("--- 开始受害者模型训练 (模式: 保存最后一个 epoch, 最后评估) ---")

    # --- Setup (Device, Seed, Data Loaders, Model, Criterion, Optimizer, Scheduler) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    random.seed(seed) # Ensure random seed is also set
    if device == torch.device("cuda"): torch.cuda.manual_seed_all(seed)

    try:
        train_loader, test_loader = get_cifar_loaders( # <<< 需要 test_loader for final eval
            dataset_name=config['dataset']['name'],
            batch_size=config['training_victim']['batch_size'],
            data_dir=config['dataset'].get('data_dir', f'./data_{config["dataset"]["name"]}'),
            augment_train=config['dataset'].get('augment_train', True)
        )
    except KeyError as e:
        print(f"数据加载失败：配置中缺少键 {e}。")
        return None
    except Exception as e:
        print(f"数据加载时发生错误: {e}")
        traceback.print_exc()
        return None

    print("初始化统一 SC 模型...")
    try:
        model = SC_Model(config=config).to(device)
        print("SC 模型初始化成功。")
    except KeyError as e:
        print(f"模型初始化失败: 配置中缺少键 {e}")
        return None
    except ValueError as e:
         print(f"模型初始化失败: 值错误 {e}")
         return None
    except Exception as e:
        print(f"初始化模型时发生意外错误: {e}")
        traceback.print_exc()
        return None

    try:
        criterion = get_victim_loss_criterion(
            config['task'],
            config['training_victim'].get('loss_type', 'default')
        ).to(device)
    except ValueError as e:
         print(f"获取损失函数失败: {e}")
         return None
    except KeyError as e:
         print(f"获取损失函数失败：配置中缺少键 {e}。")
         return None

    try:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training_victim']['lr'],
            weight_decay=config['training_victim'].get('weight_decay', 0)
        )
    except KeyError as e:
        print(f"初始化优化器失败：配置中缺少键 {e}。")
        return None


    scheduler = None # Setup scheduler as before if needed
    try:
        scheduler_config = config['training_victim'].get('lr_scheduler')
        epochs = config['training_victim']['epochs'] # Need epochs for scheduler T_max
        if scheduler_config == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            print("使用 Cosine Annealing LR 调度器。")
        elif scheduler_config == 'step':
            lr_step_size = config['training_victim'].get('lr_step_size')
            lr_gamma = config['training_victim'].get('lr_gamma')
            if lr_step_size is None or lr_gamma is None:
                print("警告: Step LR 调度器需要 'lr_step_size' 和 'lr_gamma' 配置。")
            else:
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
                print(f"使用 Step LR 调度器 (step={lr_step_size}, gamma={lr_gamma})。")
    except KeyError as e:
         print(f"初始化调度器失败：配置中缺少键 {e}。")
         return None

    # --- 初始化用于记录历史的列表 (只记录训练Loss) ---
    epochs_list = []
    train_loss_history = []

    # --- 检查点保存设置 (保存最后一个模型) ---
    checkpoint_save_dir = config.get('save_path', os.path.join(output_dir, 'checkpoints'))
    os.makedirs(checkpoint_save_dir, exist_ok=True)
    # 改动：文件名表示是最后一个 epoch 的模型
    model_filename = f'victim_{config["dataset"]["name"]}_{config["task"]}_{config["victim_model"]["type"]}_last_epoch.pth'
    last_model_path = os.path.join(checkpoint_save_dir, model_filename) # <<< 改名

    print("\n开始训练循环...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # --- 只进行训练 ---
        train_loss = train_victim_epoch(model, train_loader, optimizer, criterion, device, config['task'])

        # --- 记录训练数据 ---
        epochs_list.append(epoch)
        train_loss_history.append(train_loss)

        epoch_duration = time.time() - epoch_start_time
        # --- 移除评估相关的打印 ---
        print(f"Epoch {epoch}/{epochs} | Time: {epoch_duration:.2f}s | Train Loss: {train_loss:.4f}")

        # --- LR 调度器步骤 ---
        if scheduler:
             # 检查调度器的类型
             if isinstance(scheduler, (optim.lr_scheduler.CosineAnnealingLR, optim.lr_scheduler.StepLR)):
                 scheduler.step()
             # Add other scheduler types if needed (e.g., ReduceLROnPlateau needs a metric)

    # --- 训练循环结束 ---
    total_training_time = time.time() - start_time
    print(f"\n--- 受害者训练循环完成 ---")
    print(f"总训练时间: {total_training_time:.2f}s ({total_training_time/60:.2f} min)")

    # --- 保存最后一个 Epoch 的模型 ---
    print(f"保存最后一个 epoch ({epochs}) 的模型到 {last_model_path}")
    try:
        torch.save(model.state_dict(), last_model_path)
    except Exception as e:
        print(f"!! 保存最后一个模型时出错: {e}")
        traceback.print_exc()
        # Decide if you want to proceed without saving
        # return None # Or just print error and continue to evaluation

    # --- 执行最终评估 ---
    print("\n--- 开始最终评估 (在测试集上) ---")
    # Ensure test_loader is available
    if 'test_loader' not in locals():
         print("错误：test_loader 未定义，无法进行最终评估。")
         final_eval_results = {"error": "test_loader not available"}
    else:
        try:
             final_eval_results = evaluate_victim(model, test_loader, criterion, device, config['task'])
             print("--- 最终评估完成 ---")
             print("最终评估结果:")
             # 打印结果字典
             for key, value in final_eval_results.items():
                  if isinstance(value, float): print(f"  {key}: {value:.4f}")
                  else: print(f"  {key}: {value}")
        except Exception as e:
             print(f"!! 最终评估过程中发生错误: {e}")
             traceback.print_exc()
             final_eval_results = {"error": f"Evaluation failed: {e}"}


    # --- 保存训练历史记录 (只有训练Loss + Final Eval) ---
    training_history = {
        'epochs': epochs_list,
        'train_loss': train_loss_history,
        'final_evaluation': final_eval_results # <<< 添加最终评估结果
    }
    history_save_path = os.path.join(output_dir, 'training_history.json')
    try:
        # Convert tensors in final_eval_results to basic types if any exist
        def convert_to_basic_types(obj):
            if isinstance(obj, torch.Tensor):
                return obj.item() # Convert tensor to Python number
            elif isinstance(obj, dict):
                return {k: convert_to_basic_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_basic_types(i) for i in obj]
            return obj
        
        training_history_serializable = convert_to_basic_types(training_history)

        with open(history_save_path, 'w') as f:
            json.dump(training_history_serializable, f, indent=4)
        print(f"训练历史记录 (含最终评估) 已保存到: {history_save_path}")
    except Exception as e:
        print(f"保存训练历史记录时出错: {e}")
        traceback.print_exc()

    # --- 生成并保存训练 Loss 图表 ---
    try:
        plot_training_history(training_history, output_dir)
    except Exception as e:
        print(f"!! 生成或保存图表时出错: {e}")
        traceback.print_exc()

    # --- 返回最后一个模型的路径 ---
    return last_model_path # <<< 返回的是 last_model_path