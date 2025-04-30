# training/train_victim.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from tqdm import tqdm # For progress bars

# Assuming these modules exist in the project structure
from SC.model import SC_Model
from data.cifar import get_cifar_loaders
from evaluation.metrics import calculate_psnr, calculate_accuracy, calculate_ssim # Add SSIM later if needed
# Consider adding LPIPS if using perceptual loss: from evaluation.metrics import calculate_lpips

# --- Helper Function for Loss ---
def get_victim_loss_criterion(task: str, loss_type: str = 'default'):
    """Gets the appropriate loss function for the victim model."""
    if task == 'reconstruction':
        if loss_type == 'mse' or loss_type == 'default':
            print("Using MSE Loss for Reconstruction.")
            return nn.MSELoss()
        elif loss_type == 'l1':
            print("Using L1 Loss for Reconstruction.")
            return nn.L1Loss()
        # elif loss_type == 'perceptual':
        #     print("Using Perceptual Loss (LPIPS - requires installation).")
        #     # Make sure to install lpips: pip install lpips
        #     import lpips
        #     # Use lpips.LPIPS(net='alex') or lpips.LPIPS(net='vgg')
        #     # Note: LPIPS might require specific input ranges (e.g., [-1, 1])
        #     return lpips.LPIPS(net='vgg', spatial=True).cuda() # Example, needs device handling
        else:
            raise ValueError(f"Unsupported reconstruction loss type: {loss_type}")
    elif task == 'classification':
        # Default for classification is CrossEntropy
        print("Using CrossEntropy Loss for Classification.")
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported task for loss criterion: {task}")

# --- Training Loop ---
def train_victim_epoch(model: SC_Model,
                       loader: DataLoader,
                       optimizer: optim.Optimizer,
                       criterion: nn.Module,
                       device: torch.device,
                       task: str):
    """Runs one training epoch for the victim SC model."""
    model.train() # Set model to training mode
    total_loss = 0.0
    num_samples = 0

    progress_bar = tqdm(loader, desc=f'Training Epoch', leave=False)
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)
        num_samples += batch_size

        optimizer.zero_grad()

        # Forward pass
        # We don't necessarily need intermediate z, z' for the basic loss
        output = model(data)

        # Calculate loss based on task
        if task == 'reconstruction':
            loss = criterion(output, data) # Compare reconstructed image with original
            # Handle LPIPS case separately if needed (input format/range)
            # if isinstance(criterion, lpips.LPIPS):
            #    loss = criterion(output, data).mean() # LPIPS might return per-image loss
        elif task == 'classification':
            loss = criterion(output, target) # Compare logits with target class indices
        else:
             raise ValueError(f"Unknown task '{task}' during loss calculation")


        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size # Accumulate weighted loss

        # Update progress bar
        progress_bar.set_postfix(loss=f'{loss.item():.4f}')

    avg_loss = total_loss / num_samples
    return avg_loss


# --- Evaluation Loop ---
def evaluate_victim(model: SC_Model,
                    loader: DataLoader,
                    criterion: nn.Module, # Can use the same criterion or a different one for eval
                    device: torch.device,
                    task: str) -> dict:
    """Evaluates the victim SC model on a dataset."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    num_samples = 0
    correct_predictions = 0
    total_psnr = 0.0
    # total_ssim = 0.0 # Add later
    # total_lpips = 0.0 # Add later

    results = {}

    progress_bar = tqdm(loader, desc=f'Evaluating', leave=False)
    with torch.no_grad(): # Disable gradient calculation for evaluation
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            num_samples += batch_size

            # Forward pass
            output = model(data)

            # Calculate loss
            if task == 'reconstruction':
                loss = criterion(output, data)
                total_loss += loss.item() * batch_size
                # Calculate image quality metrics
                psnr_batch = calculate_psnr(output, data)
                total_psnr += psnr_batch * batch_size
                # ssim_batch = calculate_ssim(output, data) # Add later
                # total_ssim += ssim_batch * batch_size
                # lpips_batch = calculate_lpips(output, data) # Add later
                # total_lpips += lpips_batch * batch_size
                progress_bar.set_postfix(loss=f'{loss.item():.4f}', psnr=f'{psnr_batch:.2f}')

            elif task == 'classification':
                loss = criterion(output, target)
                total_loss += loss.item() * batch_size
                # Calculate accuracy
                acc_batch, correct_batch = calculate_accuracy(output, target)
                correct_predictions += correct_batch
                progress_bar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{acc_batch*100:.2f}%')

    # Calculate average metrics
    results['loss'] = total_loss / num_samples
    if task == 'reconstruction':
        results['psnr'] = total_psnr / num_samples
        # results['ssim'] = total_ssim / num_samples # Add later
        # results['lpips'] = total_lpips / num_samples # Add later
    elif task == 'classification':
        results['accuracy'] = correct_predictions / num_samples

    return results


# --- Main Training Orchestration ---
def train_victim_model(config: dict):
    """
    Main function to orchestrate the training of the SC victim model.

    Args:
        config (dict): A dictionary containing all necessary configurations.
                       Expected keys: 'dataset', 'task', 'victim_model', 'channel',
                                      'training_victim', 'seed', 'save_path'.
    """
    print("--- Starting Victim Model Training ---")
    print(f"Config: {config}") # Log config

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Reproducibility
    if 'seed' in config:
        torch.manual_seed(config['seed'])
        if device == torch.device("cuda"):
            torch.cuda.manual_seed_all(config['seed'])

    # Load Data
    train_loader, test_loader = get_cifar_loaders(
        dataset_name=config['dataset']['name'],
        batch_size=config['training_victim']['batch_size'],
        data_dir=config['dataset'].get('data_dir', f'./data_{config["dataset"]["name"]}'),
        augment_train=config['dataset'].get('augment_train', True)
    )
    num_classes = 10 if config['dataset']['name'] == 'cifar10' else 100

    # Initialize Model
    # Add num_classes to decoder config if task is classification
    decoder_conf = config['victim_model']['decoder'].copy() # Avoid modifying original config
    if config['task'] == 'classification':
        decoder_conf['num_classes'] = num_classes

    model = SC_Model(
        encoder_config=config['victim_model']['encoder'],
        channel_config=config['channel'],
        decoder_config=decoder_conf,
        task=config['task']
    ).to(device)

    # Loss and Optimizer
    criterion = get_victim_loss_criterion(
        config['task'],
        config['training_victim'].get('loss_type', 'default')
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training_victim']['lr'],
        weight_decay=config['training_victim'].get('weight_decay', 0) # Optional weight decay
    )

    # Optional: Learning Rate Scheduler
    scheduler = None
    if config['training_victim'].get('lr_scheduler', None) == 'cosine':
         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training_victim']['epochs'])
         print("Using Cosine Annealing LR Scheduler.")
    elif config['training_victim'].get('lr_scheduler', None) == 'step':
         scheduler = optim.lr_scheduler.StepLR(optimizer,
                                               step_size=config['training_victim']['lr_step_size'],
                                               gamma=config['training_victim']['lr_gamma'])
         print(f"Using Step LR Scheduler (step={config['training_victim']['lr_step_size']}, gamma={config['training_victim']['lr_gamma']}).")


    # Training Loop
    best_metric = -float('inf') if config['task'] == 'classification' else float('inf') # Initialize best metric
    save_path = config.get('save_path', './results/victim')
    os.makedirs(save_path, exist_ok=True)
    best_model_path = os.path.join(save_path, f'victim_model_{config["dataset"]["name"]}_{config["task"]}_best.pth')

    print("\nStarting training loop...")
    start_time = time.time()
    for epoch in range(1, config['training_victim']['epochs'] + 1):
        epoch_start_time = time.time()

        train_loss = train_victim_epoch(model, train_loader, optimizer, criterion, device, config['task'])
        eval_results = evaluate_victim(model, test_loader, criterion, device, config['task'])
        eval_loss = eval_results['loss']

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch}/{config['training_victim']['epochs']} | Time: {epoch_duration:.2f}s | "
              f"Train Loss: {train_loss:.4f} | Eval Loss: {eval_loss:.4f}", end="")

        # Log and Checkpoint Best Model based on task-specific metric
        current_metric = 0
        if config['task'] == 'reconstruction':
            current_metric = eval_results.get('psnr', -float('inf')) # Use PSNR, default to -inf if not calculated
            print(f" | Eval PSNR: {current_metric:.2f} dB")
            is_best = current_metric > best_metric # Higher PSNR is better
        elif config['task'] == 'classification':
            current_metric = eval_results.get('accuracy', 0.0)
            print(f" | Eval Accuracy: {current_metric*100:.2f}%")
            is_best = current_metric > best_metric # Higher Accuracy is better

        if is_best:
            best_metric = current_metric
            print(f"  => Saving new best model to {best_model_path}")
            torch.save(model.state_dict(), best_model_path)
        else:
             print("") # Newline if not saving


        # LR Scheduler Step
        if scheduler:
             scheduler.step()

    total_training_time = time.time() - start_time
    print(f"\n--- Victim Training Finished ---")
    print(f"Total Training Time: {total_training_time:.2f}s")
    print(f"Best Evaluation Metric ({'Accuracy' if config['task'] == 'classification' else 'PSNR'}): {best_metric:.4f}")
    print(f"Best model saved to: {best_model_path}")

    # Return the path to the best model for potential use by the attacker setup
    return best_model_path


# Example configuration (can be loaded from YAML or defined here)
if __name__ == '__main__':
    dummy_config_recon = {
        'dataset': {'name': 'cifar10', 'data_dir': './data_cifar10'},
        'task': 'reconstruction',
        'victim_model': {
            'encoder': {'arch_name': 'resnet18', 'latent_dim': 32, 'pretrained': False},
            'decoder': {'arch_name': 'resnet18', 'latent_dim': 32}
        },
        'channel': {'type': 'awgn', 'snr_db': 15},
        'training_victim': {
            'batch_size': 128,
            'lr': 1e-3,
            'epochs': 5, # Keep epochs low for quick test
            'loss_type': 'mse'
        },
        'seed': 42,
        'save_path': './results/test_victim_recon'
    }

    dummy_config_classify = {
         'dataset': {'name': 'cifar10', 'data_dir': './data_cifar10'},
         'task': 'classification',
         'victim_model': {
             'encoder': {'arch_name': 'resnet18', 'latent_dim': 64, 'pretrained': False},
             'decoder': {'arch_name': 'resnet18', 'latent_dim': 64, 'dropout': 0.5} # num_classes added in train func
         },
         'channel': {'type': 'ideal'}, # Ideal channel for simple classification test
         'training_victim': {
             'batch_size': 128,
             'lr': 1e-3,
             'epochs': 5,
             'lr_scheduler': 'step', # Example scheduler
             'lr_step_size': 3,
             'lr_gamma': 0.1
         },
         'seed': 42,
         'save_path': './results/test_victim_classify'
     }

    print("===== Testing Reconstruction Training =====")
    train_victim_model(dummy_config_recon)

    print("\n===== Testing Classification Training =====")
    train_victim_model(dummy_config_classify)