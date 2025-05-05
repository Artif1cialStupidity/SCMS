# MS/train_surrogate.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
# Remove VictimQueryInterface import if not needed directly here
# from MS.query_interface import VictimQueryInterface
import torch.nn.functional as F # For loss and similarity metrics

# train_surrogate_epoch remains largely the same, ensure it takes 'model' argument
def train_surrogate_epoch(model: nn.Module, # Changed arg name to generic 'model'
                          loader: DataLoader,
                          optimizer: optim.Optimizer,
                          criterion: nn.Module,
                          device: torch.device,
                          attack_type: str, # Keep for potential logging
                          task: str = 'latent_matching'): # Default or specific task hint
    """
    Runs one training epoch for the attacker's surrogate model (e.g., encoder).
    Args:
        model (nn.Module): The surrogate model being trained (e.g., encoder).
        # ... other args as before ...
    Returns:
        float: Average loss for the epoch.
    """
    model.train() # Set surrogate model to training mode
    total_loss = 0.0
    num_samples = 0

    progress_bar = tqdm(loader, desc=f'Training Surrogate ({task})', leave=False)
    for batch_idx, (input_data, target_data) in enumerate(progress_bar):
        input_data = input_data.to(device) # This is X
        target_data = target_data.to(device) # This is the target z_observed
        batch_size = input_data.size(0)
        num_samples += batch_size

        optimizer.zero_grad()

        # Forward pass through the surrogate model (encoder)
        surrogate_output = model(input_data) # surrogate_output is surrogate_z

        # Calculate loss: surrogate_output (surrogate_z) vs victim's observed z (target_data)
        loss = criterion(surrogate_output, target_data)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        progress_bar.set_postfix(loss=f'{loss.item():.6f}')

    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    return avg_loss

# --- Updated Fidelity Evaluation Function (Encoder vs Encoder) ---

def evaluate_surrogate_fidelity(victim_encoder: nn.Module,
                                surrogate_encoder: nn.Module,
                                test_loader: DataLoader,
                                device: torch.device) -> dict:
    """
    Evaluates how well the surrogate ENCODER mimics the victim ENCODER on a test set.
    Compares their latent outputs (z) for the same input X.

    Args:
        victim_encoder (nn.Module): The original victim encoder model.
        surrogate_encoder (nn.Module): The trained surrogate encoder model.
        test_loader (DataLoader): DataLoader for the original test dataset (provides X).
        device (torch.device): Device to run evaluation on.

    Returns:
        dict: Dictionary containing fidelity metrics (e.g., 'latent_mse', 'latent_cosine_similarity').
    """
    victim_encoder.eval()
    surrogate_encoder.eval()

    all_victim_z = []
    all_surrogate_z = []
    total_samples = 0

    progress_bar = tqdm(test_loader, desc="Evaluating Encoder Fidelity", leave=False)
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(progress_bar): # Input data X, ignore labels
            data = data.to(device)
            batch_size = data.size(0)
            total_samples += batch_size

            # Get latent vectors from both encoders for the same input X
            victim_z = victim_encoder(data)
            surrogate_z = surrogate_encoder(data)

            all_victim_z.append(victim_z.cpu())
            all_surrogate_z.append(surrogate_z.cpu())

    results = {}
    if not all_victim_z:
        print("Warning: No data processed for fidelity evaluation.")
        return results

    victim_z_all = torch.cat(all_victim_z, dim=0)
    surrogate_z_all = torch.cat(all_surrogate_z, dim=0)

    # Calculate metrics comparing the latent vectors z
    # 1. Mean Squared Error (MSE) between latent vectors
    latent_mse = F.mse_loss(surrogate_z_all, victim_z_all).item()
    results['latent_mse'] = latent_mse

    # 2. Cosine Similarity between latent vectors
    # Ensure vectors are not zero vectors before calculating cosine similarity
    epsilon = 1e-8
    victim_norm = victim_z_all.norm(p=2, dim=1, keepdim=True)
    surrogate_norm = surrogate_z_all.norm(p=2, dim=1, keepdim=True)

    # Avoid division by zero for zero vectors, set their similarity to 0 or 1? Let's use 0.
    valid_mask = (victim_norm > epsilon) & (surrogate_norm > epsilon)
    valid_mask = valid_mask.squeeze()

    if valid_mask.any():
         cosine_sim = F.cosine_similarity(surrogate_z_all[valid_mask], victim_z_all[valid_mask], dim=1)
         latent_cosine_sim_mean = cosine_sim.mean().item()
    else:
         latent_cosine_sim_mean = 0.0 # Or nan?

    results['latent_cosine_similarity'] = latent_cosine_sim_mean

    # Can add L1 loss if desired
    # latent_l1 = F.l1_loss(surrogate_z_all, victim_z_all).item()
    # results['latent_l1'] = latent_l1

    return results