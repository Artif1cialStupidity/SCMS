# train_sc_model.py

import torch
import yaml
import argparse
import os
import pprint
import random

# Adjust import paths based on the structure
from SC.train_victim import train_victim_model # Import the training orchestrator function
from config.config_utils import load_config # Assuming you might create a config_utils.py later

# --- Argument Parsing ---
def parse_args():
    """Parses command line arguments for victim training."""
    parser = argparse.ArgumentParser(description="Train the Semantic Communication (Victim) Model")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the configuration YAML file for the victim model.')
    # Output directory for the *results* of this training run (logs, best model)
    parser.add_argument('--output-dir', type=str, default='./results/victim_training_run',
                        help='Directory to save training results and the best model checkpoint.')
    args = parser.parse_args()
    return args

# --- Main Function ---
def main():
    args = parse_args()

    # --- Load Config ---
    # Assuming load_config handles file not found etc.
    config = load_config(args.config)
    if config is None:
        return # Exit if config loading failed

    # --- Setup ---
    # Create output directory
    # The train_victim_model function will handle saving within a subfolder possibly defined in config
    os.makedirs(args.output_dir, exist_ok=True)
    config['training_victim']['save_path'] = os.path.join(args.output_dir, 'victim_checkpoint') # Standardize save location
    print(f"Victim model will be saved under: {config['training_victim']['save_path']}")


    print("\n--- Victim Training Configuration ---")
    pprint.pprint(config, indent=2)
    print("-------------------------------------\n")

    # Save config copy
    config_save_path = os.path.join(args.output_dir, 'victim_config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Set device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    random.seed(seed)
    if device == torch.device("cuda"):
        torch.cuda.manual_seed_all(seed)

    # --- Start Training ---
    print("Starting SC Victim Model Training...")
    # The train_victim_model function handles data loading, model init, training loop, eval, saving
    best_model_path = train_victim_model(config)

    if best_model_path and os.path.exists(best_model_path):
        print(f"\nTraining complete. Best victim model saved to: {best_model_path}")
    else:
        print("\nTraining complete, but couldn't verify the saved model path.")

if __name__ == '__main__':
     # Simple helper function for loading config (can be moved to utils)
     def load_config(config_path):
         try:
             with open(config_path, 'r') as f:
                 config = yaml.safe_load(f)
             print(f"Loaded configuration from: {config_path}")
             return config
         except FileNotFoundError:
             print(f"Error: Configuration file not found at {config_path}")
             return None
         except yaml.YAMLError as e:
             print(f"Error parsing configuration file: {e}")
             return None

     main()