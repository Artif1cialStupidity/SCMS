# SC/train_victim.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from tqdm import tqdm # 用于进度条

# --- 更新导入 ---
# 从新的 models 目录导入统一的 SC_Model
from models.model import SC_Model
# 数据加载和评估指标的导入保持不变
from data.cifar import get_cifar_loaders
from evaluation.metrics import calculate_psnr, calculate_accuracy, calculate_ssim # SSIM 可能仍需要安装 scikit-image

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
        # 可以根据需要添加其他损失，例如 perceptual loss (LPIPS)
        # elif loss_type.lower() == 'perceptual':
        #     # import lpips
        #     # return lpips.LPIPS(...)
        else:
            raise ValueError(f"不支持的重建损失类型: {loss_type}")
    elif task == 'classification':
        print("使用 CrossEntropy Loss 进行分类。")
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"不支持的任务以获取损失标准: {task}")

# --- 训练循环 ---
def train_victim_epoch(model: SC_Model, # 类型提示更新为统一模型
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

# --- 评估循环 ---
def evaluate_victim(model: SC_Model, # 类型提示更新为统一模型
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
    # total_ssim = 0.0 # 如果需要计算 SSIM

    results = {}

    progress_bar = tqdm(loader, desc=f'评估中', leave=False)
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
                # 计算图像质量指标
                # 注意：确保 calculate_psnr 接收的数据范围正确（例如，如果Tanh输出[-1,1]，则data_range=2.0）
                psnr_batch = calculate_psnr(output, data, data_range=2.0) # 假设数据范围是 [-1, 1]
                total_psnr += psnr_batch * batch_size
                # ssim_batch = calculate_ssim(output, data) # 如果使用
                # total_ssim += ssim_batch * batch_size
                progress_bar.set_postfix(loss=f'{loss.item():.4f}', psnr=f'{psnr_batch:.2f}')

            elif task == 'classification':
                loss = criterion(output, target)
                total_loss += loss.item() * batch_size
                # 计算准确率
                acc_batch, correct_batch = calculate_accuracy(output, target)
                correct_predictions += correct_batch
                progress_bar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{acc_batch*100:.2f}%')

    # 计算平均指标
    results['loss'] = total_loss / num_samples if num_samples > 0 else 0.0
    if task == 'reconstruction':
        results['psnr'] = total_psnr / num_samples if num_samples > 0 else 0.0
        # results['ssim'] = total_ssim / num_samples # 如果使用
    elif task == 'classification':
        results['accuracy'] = correct_predictions / num_samples if num_samples > 0 else 0.0

    return results

# --- 主要训练协调函数 ---
def train_victim_model(config: dict):
    """
    协调 SC 受害者模型训练的主函数。

    Args:
        config (dict): 包含所有必要配置的字典。
                       预期键: 'dataset', 'task', 'victim_model', 'channel',
                               'training_victim', 'seed', 'save_path'.
                       'victim_model' 必须包含 'type'。
    """
    print("--- 开始受害者模型训练 ---")
    # print(f"配置: {config}") # 打印配置用于调试

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 可复现性
    seed = config.get('seed', 42) # 从配置获取种子
    torch.manual_seed(seed)
    if device == torch.device("cuda"):
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True # 可能影响性能
        # torch.backends.cudnn.benchmark = False

    # 加载数据
    train_loader, test_loader = get_cifar_loaders(
        dataset_name=config['dataset']['name'],
        batch_size=config['training_victim']['batch_size'],
        data_dir=config['dataset'].get('data_dir', f'./data_{config["dataset"]["name"]}'), # 使用配置中的路径
        augment_train=config['dataset'].get('augment_train', True)
    )
    # 注意：类别数现在由 SC_Model 的 __init__ 内部根据数据集名称处理

    # --- 初始化模型 (使用统一的 SC_Model 和完整配置) ---
    print("初始化统一 SC 模型...")
    try:
        # 将整个配置字典传递给模型构造函数
        model = SC_Model(config=config).to(device)
        print("SC 模型初始化成功。")
    except (KeyError, ValueError) as e:
        print(f"模型初始化失败: {e}")
        print("请检查配置文件是否包含所有必需的键，特别是 'victim_model.type'。")
        return None # 初始化失败则返回
    except Exception as e:
        print(f"初始化模型时发生意外错误: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 损失函数和优化器
    criterion = get_victim_loss_criterion(
        config['task'],
        config['training_victim'].get('loss_type', 'default') # 从配置获取损失类型
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training_victim']['lr'],
        weight_decay=config['training_victim'].get('weight_decay', 0) # 从配置获取权重衰减
    )

    # 可选：学习率调度器
    scheduler = None
    scheduler_config = config['training_victim'].get('lr_scheduler') # 简化获取
    if scheduler_config == 'cosine':
         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training_victim']['epochs'])
         print("使用 Cosine Annealing LR 调度器。")
    elif scheduler_config == 'step':
         lr_step_size = config['training_victim'].get('lr_step_size')
         lr_gamma = config['training_victim'].get('lr_gamma')
         if lr_step_size is None or lr_gamma is None:
             print("警告: Step LR 调度器需要 'lr_step_size' 和 'lr_gamma' 配置。")
         else:
             scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
             print(f"使用 Step LR 调度器 (step={lr_step_size}, gamma={lr_gamma})。")

    # 训练循环
    best_metric = -float('inf') if config['task'] == 'reconstruction' else 0.0 # PSNR 越高越好, Acc 越高越好
    save_path_dir = config.get('save_path', './results/victim_models') # 使用配置或默认值
    os.makedirs(save_path_dir, exist_ok=True)
    # 文件名可以包含更多信息
    model_filename = f'victim_{config["dataset"]["name"]}_{config["task"]}_{config["victim_model"]["type"]}_best.pth'
    best_model_path = os.path.join(save_path_dir, model_filename)

    print("\n开始训练循环...")
    start_time = time.time()
    epochs = config['training_victim']['epochs']
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        train_loss = train_victim_epoch(model, train_loader, optimizer, criterion, device, config['task'])
        eval_results = evaluate_victim(model, test_loader, criterion, device, config['task'])
        eval_loss = eval_results['loss']

        epoch_duration = time.time() - epoch_start_time
        log_prefix = f"Epoch {epoch}/{epochs} | Time: {epoch_duration:.2f}s | Train Loss: {train_loss:.4f} | Eval Loss: {eval_loss:.4f}"

        # 记录和检查点最佳模型
        current_metric = 0
        metric_name = ""
        is_best = False
        if config['task'] == 'reconstruction':
            metric_name = 'PSNR'
            current_metric = eval_results.get('psnr', -float('inf'))
            print(f"{log_prefix} | Eval PSNR: {current_metric:.2f} dB")
            is_best = current_metric > best_metric # PSNR 越高越好
        elif config['task'] == 'classification':
            metric_name = 'Accuracy'
            current_metric = eval_results.get('accuracy', 0.0)
            print(f"{log_prefix} | Eval Accuracy: {current_metric*100:.2f}%")
            is_best = current_metric > best_metric # Acc 越高越好

        if is_best:
            best_metric = current_metric
            print(f"  => 新的最佳 {metric_name} ({best_metric:.4f}). 保存模型到 {best_model_path}")
            try:
                torch.save(model.state_dict(), best_model_path)
            except Exception as e:
                print(f"  !! 保存模型时出错: {e}")
        # else:
        #      print("") # 如果不保存，则打印空行以保持格式

        # LR 调度器步骤
        if scheduler:
             # 检查调度器的类型，有些可能需要在 epoch 结束时 step，有些在 batch 结束时
             if isinstance(scheduler, (optim.lr_scheduler.CosineAnnealingLR, optim.lr_scheduler.StepLR)):
                 scheduler.step()
             # elif isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
             #     scheduler.step(eval_loss) # 例如，基于验证损失调整

    total_training_time = time.time() - start_time
    print(f"\n--- 受害者训练完成 ---")
    print(f"总训练时间: {total_training_time:.2f}s ({total_training_time/60:.2f} min)")
    print(f"最佳评估指标 ({metric_name}): {best_metric:.4f}")
    print(f"最佳模型已保存到: {best_model_path}")

    # 返回最佳模型的路径，供后续步骤（如攻击）使用
    return best_model_path