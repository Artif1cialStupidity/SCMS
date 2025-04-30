# training/train_attacker.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm # For progress bars

def train_surrogate_epoch(surrogate_model: nn.Module,
                          loader: DataLoader,
                          optimizer: optim.Optimizer,
                          criterion: nn.Module,
                          device: torch.device,
                          attack_type: str,
                          task: str = 'reconstruction'): # Task needed for potential KLDiv loss case
    """
    Runs one training epoch for the attacker's surrogate model.

    Args:
        surrogate_model (nn.Module): The surrogate model being trained.
        loader (DataLoader): DataLoader providing the (input, target) pairs
                             collected from the victim.
        optimizer (optim.Optimizer): Optimizer for the surrogate model.
        criterion (nn.Module): Loss function to compare surrogate output with victim output.
                               (e.g., MSE for latent vectors or logits, maybe KLDiv).
        device (torch.device): Device to run training on.
        attack_type (str): Type of attack ('steal_encoder', 'steal_decoder', 'steal_end2end').
                           Potentially useful if loss logic differs slightly.
        task (str): Victim's task ('reconstruction' or 'classification'). Needed if using
                    KL divergence loss for classification logits.

    Returns:
        float: Average loss for the epoch.
    """
    surrogate_model.train() # Set surrogate model to training mode
    total_loss = 0.0
    num_samples = 0

    progress_bar = tqdm(loader, desc=f'Training Surrogate', leave=False)
    for batch_idx, (input_data, target_data) in enumerate(progress_bar):
        input_data = input_data.to(device)
        target_data = target_data.to(device) # This is the victim's output (z, y, etc.)
        batch_size = input_data.size(0)
        num_samples += batch_size

        optimizer.zero_grad()

        # Forward pass through the surrogate model
        surrogate_output = surrogate_model(input_data)

        # Calculate loss: surrogate_output vs victim's output (target_data)
        # Special handling for KL Divergence if used for classification logits:
        # if isinstance(criterion, nn.KLDivLoss) and task == 'classification' and \
        #    (attack_type == 'steal_decoder' or attack_type == 'steal_end2end'):
        #      # Requires victim output (target_data) to be probabilities (apply softmax)
        #      # and surrogate output to be log-probabilities (apply log_softmax)
        #      surrogate_log_probs = F.log_softmax(surrogate_output, dim=1)
        #      victim_probs = F.softmax(target_data, dim=1) # Assuming target_data are logits
        #      loss = criterion(surrogate_log_probs, victim_probs)
        # else:
             # Default case (e.g., MSE loss for latent vectors or logits)
             loss = criterion(surrogate_output, target_data)


        loss.backward()
        # Optional: Gradient clipping can sometimes help stabilize training
        # torch.nn.utils.clip_grad_norm_(surrogate_model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch_size # Accumulate weighted loss

        # Update progress bar
        progress_bar.set_postfix(loss=f'{loss.item():.4f}')

    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    return avg_loss

# --- Fidelity Evaluation Function ---
# This function compares the surrogate and victim models directly.
# It could live here or in attacker/attacker.py or evaluation/metrics.py.
# Let's put it here for now as it's closely related to the surrogate training goal.

def evaluate_surrogate_fidelity(victim_interface: VictimQueryInterface,
                                surrogate_model: nn.Module,
                                test_loader: DataLoader,
                                device: torch.device,
                                task: str,
                                attack_type: str,
                                latent_access: str) -> dict:
    """
    Evaluates how well the surrogate model mimics the victim model on a test set.

    Args:
        victim_interface (VictimQueryInterface): Interface to query the victim.
        surrogate_model (nn.Module): The trained surrogate model.
        test_loader (DataLoader): DataLoader for the original test dataset.
        device (torch.device): Device to run evaluation on.
        task (str): The victim's task ('reconstruction' or 'classification').
        attack_type (str): The type of attack performed.
        latent_access (str): The latent access level the attacker had.

    Returns:
        dict: Dictionary containing fidelity metrics (e.g., 'model_agreement', 'output_mse').
    """
    victim_interface.victim_model.eval() # Ensure victim is in eval mode
    surrogate_model.eval() # Ensure surrogate is in eval mode

    all_victim_outputs = []
    all_surrogate_outputs = []
    total_samples = 0

    progress_bar = tqdm(test_loader, desc="Evaluating Fidelity", leave=False)
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(progress_bar): # Ignore original labels here
            data = data.to(device)
            batch_size = data.size(0)
            total_samples += batch_size

            # Determine the input for the surrogate and how to get the comparable victim output
            surrogate_input = None
            victim_target_output = None # The victim output the surrogate should match

            if attack_type == 'steal_encoder':
                surrogate_input = data
                # Need victim's clean z
                victim_output_package = victim_interface.query(data) # Query using current interface settings
                if latent_access == 'clean_z' or victim_interface.query_access == 'encoder_query':
                     # If access gives clean_z, use it directly
                     if isinstance(victim_output_package, tuple): victim_target_output = victim_output_package[1]
                     else: victim_target_output = victim_output_package # Primary output is z
                else:
                     # Cannot directly evaluate fidelity if clean_z wasn't available during attack training phase
                     print("Warning: Cannot evaluate steal_encoder fidelity without clean_z access during evaluation.")
                     continue # Skip this batch for fidelity calc
                surrogate_output = surrogate_model(surrogate_input)

            elif attack_type == 'steal_decoder':
                 # Surrogate input is z'. Need victim's Y for the same z'.
                 # Generate z' using victim's encoder + channel
                 z = victim_interface.victim_model.encode(data)
                 victim_interface.victim_model.channel.eval()
                 z_prime = victim_interface.victim_model.channel(z)
                 surrogate_input = z_prime
                 # Get victim's output for this z_prime
                 victim_target_output = victim_interface.victim_model.decode(z_prime)
                 surrogate_output = surrogate_model(surrogate_input)

            elif attack_type == 'steal_end2end':
                 surrogate_input = data
                 # Need victim's Y for this X
                 victim_output_package = victim_interface.query(data) # Query using interface settings
                 if isinstance(victim_output_package, tuple): victim_target_output = victim_output_package[0] # Primary output is Y
                 else: victim_target_output = victim_output_package
                 surrogate_output = surrogate_model(surrogate_input)

            else:
                 raise ValueError(f"Unknown attack type for fidelity evaluation: {attack_type}")


            if victim_target_output is not None:
                 all_victim_outputs.append(victim_target_output.cpu())
                 all_surrogate_outputs.append(surrogate_output.cpu())


    results = {}
    if not all_victim_outputs: # Check if we skipped all batches
        print("Warning: No data collected for fidelity evaluation.")
        return results

    victim_outputs_all = torch.cat(all_victim_outputs, dim=0)
    surrogate_outputs_all = torch.cat(all_surrogate_outputs, dim=0)

    # Calculate metrics based on task and attack type
    if task == 'classification' and (attack_type == 'steal_decoder' or attack_type == 'steal_end2end'):
        agreement = calculate_model_agreement(victim_outputs_all, surrogate_outputs_all, task='classification')
        results['model_agreement'] = agreement
        # Also calculate MSE/L1 on logits as a fidelity measure
        logit_mse = F.mse_loss(surrogate_outputs_all, victim_outputs_all).item()
        results['logit_mse'] = logit_mse
    elif task == 'reconstruction' and (attack_type == 'steal_decoder' or attack_type == 'steal_end2end'):
        # Compare reconstructed images
        output_mse = F.mse_loss(surrogate_outputs_all, victim_outputs_all).item()
        results['output_mse'] = output_mse
        # Could also calculate PSNR/SSIM/LPIPS between victim/surrogate outputs
        fidelity_psnr = calculate_psnr(surrogate_outputs_all, victim_outputs_all, data_range=2.0) # Assuming [-1,1] range
        results['fidelity_psnr'] = fidelity_psnr
    elif attack_type == 'steal_encoder':
        # Compare latent vectors
        latent_mse = F.mse_loss(surrogate_outputs_all, victim_outputs_all).item()
        results['latent_mse'] = latent_mse
        # Could calculate cosine similarity too
        latent_cosine_sim = F.cosine_similarity(surrogate_outputs_all, victim_outputs_all, dim=1).mean().item()
        results['latent_cosine_similarity'] = latent_cosine_sim

    return results


# Example usage (placeholder, real usage is within attacker.py)
if __name__ == '__main__':
    print("This file contains helper functions for training the surrogate model.")
    print("Run attacker.py to execute the attack and use these functions.")
    # You could add specific unit tests for train_surrogate_epoch here if desired,
    # creating dummy models, loaders, and criteria.