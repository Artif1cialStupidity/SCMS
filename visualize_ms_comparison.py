# visualize_ms_comparison.py

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import yaml
import argparse
import os
import numpy as np
import sys
import traceback

# --- Imports ---
try:
    # Victim Model
    from models.model import SC_Model
    # Surrogate Encoder Model (Need its definition, e.g., ResNetEncoderSC)
    from models.components.resnet_components import ResNetEncoderSC
    # Add other surrogate encoder types if needed (e.g., BaseTransmitter)
    from models.components.base_components import BaseTransmitter
    # Data utilities
    from data.cifar import get_cifar_loaders, CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD
    # Config utilities
    from config.config_utils import load_config
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure 'models', 'data', 'config' packages are accessible.")
    sys.exit(1)

def parse_args():
    """Parses command line arguments for visualization."""
    parser = argparse.ArgumentParser(description="Visualize Original vs Victim Output vs Surrogate Encoder Output")
    # Victim info
    parser.add_argument('--victim-checkpoint', type=str, required=True,
                        help='Path to the pre-trained victim SC model checkpoint (.pth).')
    parser.add_argument('--victim-config', type=str, required=True,
                        help='Path to the victim model configuration YAML.')
    # Surrogate info
    parser.add_argument('--surrogate-checkpoint', type=str, required=True,
                        help='Path to the trained surrogate ENCODER checkpoint (.pth).')
    parser.add_argument('--attack-config', type=str, required=True,
                        help='Path to the attack configuration YAML (needed for surrogate architecture).')
    # Visualization options
    parser.add_argument('--num-images', type=int, default=5,
                        help='Number of images to display.')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Optional: Path to save the visualization image.')
    args = parser.parse_args()
    return args

def denormalize(tensor, mean, std):
    """Denormalizes image tensor back to [0, 1] range for display."""
    tensor = tensor.clone().cpu()
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    tensor = tensor.permute(1, 2, 0)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.numpy()

def get_surrogate_encoder_instance(attack_config: dict) -> nn.Module:
    """Instantiates the surrogate encoder based on attack config."""
    if 'attacker' not in attack_config or 'surrogate_model' not in attack_config['attacker']:
        raise ValueError("Attack config must contain 'attacker.surrogate_model' section.")

    surrogate_config = attack_config['attacker']['surrogate_model']
    arch = surrogate_config.get('arch_name')
    latent_dim = surrogate_config.get('latent_dim')

    if not arch or not latent_dim:
        raise ValueError("Surrogate config must specify 'arch_name' and 'latent_dim'.")

    print(f"Instantiating surrogate encoder: Arch={arch}, LatentDim={latent_dim}")
    # Add more architectures here if the attacker supports them
    if 'resnet' in arch.lower():
        # Assuming pretrained=False for surrogate loaded from checkpoint
        return ResNetEncoderSC(arch_name=arch, latent_dim=latent_dim, pretrained=False)
    elif 'base' in arch.lower(): # Example if BaseTransmitter was used
         # Assuming input_channels=3 for CIFAR
         return BaseTransmitter(input_channels=3, latent_dim=latent_dim)
    else:
        raise ValueError(f"Unsupported surrogate encoder architecture '{arch}' specified in attack config.")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Configs ---
    print("Loading configurations...")
    victim_config = load_config(args.victim_config)
    attack_config_full = load_config(args.attack_config)
    if victim_config is None or attack_config_full is None:
        print("Error loading config files.")
        return

    victim_task = victim_config.get('task', 'unknown').lower()
    if victim_task != 'reconstruction':
        print(f"Warning: Visualization is designed for reconstruction tasks. Victim task is '{victim_task}'. "
              "Output comparison might not be meaningful.")

    # --- Load Victim Model ---
    print("Loading victim model...")
    try:
        victim_model = SC_Model(config=victim_config).to(device)
        victim_model.load_state_dict(torch.load(args.victim_checkpoint, map_location=device))
        victim_model.eval()
        print("Victim model loaded successfully.")
    except Exception as e:
        print(f"Error loading victim model: {e}")
        traceback.print_exc()
        return

    # --- Load Surrogate Encoder Model ---
    print("Loading surrogate encoder model...")
    try:
        # Instantiate based on attack config
        surrogate_encoder = get_surrogate_encoder_instance(attack_config_full).to(device)
        # Load state dict
        surrogate_encoder.load_state_dict(torch.load(args.surrogate_checkpoint, map_location=device))
        surrogate_encoder.eval()
        print("Surrogate encoder loaded successfully.")
    except Exception as e:
        print(f"Error loading surrogate encoder: {e}")
        traceback.print_exc()
        return

    # --- Load Data ---
    dataset_name = victim_config['dataset']['name']
    data_dir = victim_config['dataset'].get('data_dir', f'./data_{dataset_name}')
    print(f"Loading dataset: {dataset_name}")
    try:
        _, test_loader = get_cifar_loaders(
            dataset_name=dataset_name,
            batch_size=max(args.num_images, 10), # Get a small batch
            data_dir=data_dir,
            augment_train=False, num_workers=0
        )
        data_iter = iter(test_loader)
        data_batch, _ = next(data_iter)
        if data_batch.size(0) < args.num_images:
            print(f"Warning: Test batch size ({data_batch.size(0)}) < requested images ({args.num_images}). Using {data_batch.size(0)}.")
            args.num_images = data_batch.size(0)
        sample_x = data_batch[:args.num_images].to(device)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- Perform Inference ---
    print("Performing inference...")
    with torch.no_grad():
        # Path 1: Victim Full Path
        y_victim, z_victim, z_prime_victim = victim_model(sample_x, return_latent=True)

        # Path 2: Surrogate Encoder + Victim Channel/Decoder
        z_surrogate = surrogate_encoder(sample_x)
        # Use victim's channel and decoder
        victim_model.channel.eval() # Ensure channel is eval mode
        victim_model.decoder.eval() # Ensure decoder is eval mode
        z_prime_surrogate = victim_model.channel(z_surrogate)
        y_surrogate = victim_model.decoder(z_prime_surrogate)


    # --- Prepare for Visualization ---
    sample_x_cpu = sample_x.cpu()
    y_victim_cpu = y_victim.cpu()
    y_surrogate_cpu = y_surrogate.cpu()

    if dataset_name == 'cifar10': mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif dataset_name == 'cifar100': mean, std = CIFAR100_MEAN, CIFAR100_STD
    else: mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) # Default fallback

    # --- Plotting ---
    print("Generating visualization...")
    num_cols = 3 # Original | Victim Output | Surrogate Output
    fig, axs = plt.subplots(args.num_images, num_cols, figsize=(num_cols * 3, 2.5 * args.num_images))
    if args.num_images == 1: axs = np.array([axs]) # Ensure axs is indexable

    fig.suptitle('Original vs. Victim vs. Surrogate Encoder Reconstruction', fontsize=14)

    for i in range(args.num_images):
        # Original
        img_orig = denormalize(sample_x_cpu[i], mean, std)
        axs[i, 0].imshow(img_orig)
        axs[i, 0].set_title(f"Original #{i+1}")
        axs[i, 0].axis('off')

        # Victim Output
        if victim_task == 'reconstruction':
            img_victim = denormalize(y_victim_cpu[i], mean, std)
            axs[i, 1].imshow(img_victim)
            axs[i, 1].set_title(f"Victim Output #{i+1}")
        else: # Handle non-reconstruction task display
             axs[i, 1].text(0.5, 0.5, f'Victim Task:\n{victim_task.capitalize()}', ha='center', va='center', fontsize=9)
             axs[i, 1].set_title(f"Victim Output #{i+1}")
        axs[i, 1].axis('off')

        # Surrogate Output (using victim channel/decoder)
        if victim_task == 'reconstruction':
            img_surrogate = denormalize(y_surrogate_cpu[i], mean, std)
            axs[i, 2].imshow(img_surrogate)
            axs[i, 2].set_title(f"Surrogate Enc Output #{i+1}")
        else: # Handle non-reconstruction task display
             axs[i, 2].text(0.5, 0.5, f'Surrogate Enc\n+ Victim Dec ({victim_task.capitalize()})', ha='center', va='center', fontsize=9)
             axs[i, 2].set_title(f"Surrogate Enc Output #{i+1}")
        axs[i, 2].axis('off')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

    # Save or Show
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        try:
            plt.savefig(args.output_file, dpi=150)
            print(f"Visualization saved to: {args.output_file}")
        except Exception as e:
            print(f"Error saving visualization: {e}")
    else:
        print("Displaying visualization...")
        plt.show()

if __name__ == '__main__':
    main()