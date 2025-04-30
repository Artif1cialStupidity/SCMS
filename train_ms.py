# run_ms_attack.py

import torch
import yaml
import argparse
import os
import pprint
import random
import json

# Adjust import paths
from data.cifar import get_cifar_loaders
from SC.model import SC_Model # Need the victim model class definition
from MS.query_interface import VictimQueryInterface
from MS.attacker import ModelStealingAttacker
from config.config_utils import load_config # Assuming config_utils.py

# --- Argument Parsing ---
def parse_args():
    """Parses command line arguments for running the attack."""
    parser = argparse.ArgumentParser(description="Run Model Stealing Attack on a pre-trained SC Model")
    parser.add_argument('--attack-config', type=str, required=True,
                        help='Path to the configuration YAML file for the attack.')
    parser.add_argument('--victim-checkpoint', type=str, required=True,
                        help='Path to the pre-trained victim SC model checkpoint (.pth file).')
    # Note: Victim's original config might also be needed to correctly initialize its structure
    parser.add_argument('--victim-config', type=str, required=True,
                        help='Path to the original configuration YAML file used to train the victim model.')
    parser.add_argument('--output-dir', type=str, default='./results/attack_run',
                        help='Directory to save attack results (surrogate model, logs, metrics).')
    args = parser.parse_args()
    return args

# --- Main Function ---
def main():
    args = parse_args()

    # --- Load Configs ---
    attack_config = load_config(args.attack_config)
    victim_config = load_config(args.victim_config) # Need victim's config to init its structure
    if attack_config is None or victim_config is None:
        return # Exit if config loading failed

    # --- Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Attack results will be saved to: {args.output_dir}")

    print("\n--- Attack Configuration ---")
    pprint.pprint(attack_config, indent=2)
    print("----------------------------\n")
    # Save configs
    with open(os.path.join(args.output_dir, 'attack_config.yaml'), 'w') as f:
         yaml.dump(attack_config, f, default_flow_style=False)
    with open(os.path.join(args.output_dir, 'victim_config_used.yaml'), 'w') as f:
          yaml.dump(victim_config, f, default_flow_style=False)


    # Set device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    seed = attack_config.get('seed', 43) # Use different default seed maybe
    torch.manual_seed(seed)
    random.seed(seed)
    if device == torch.device("cuda"):
        torch.cuda.manual_seed_all(seed)

    # --- Load Data (Test set for evaluation, Proxy set for attacker) ---
    # Assuming both configs define the dataset similarly, or use attacker's definition
    dataset_name = victim_config['dataset']['name'] # Use victim's dataset name
    data_dir = victim_config['dataset'].get('data_dir', f'./data_{dataset_name}')
    num_classes = 10 if dataset_name == 'cifar10' else 100

    # Test loader for final evaluation
    # Use a reasonable batch size for evaluation
    eval_batch_size = attack_config['attacker']['training_attacker'].get('batch_size', 128)
    _, test_loader = get_cifar_loaders(
        dataset_name=dataset_name,
        batch_size=eval_batch_size,
        data_dir=data_dir,
        augment_train=False # No augmentation for testing
    )
    # Proxy loader for attacker queries (using test set for simplicity)
    proxy_loader_batch_size = attack_config['attacker']['training_attacker'].get('query_batch_size', 128)
    _, proxy_loader = get_cifar_loaders(
         dataset_name=dataset_name, # Or use a different dataset specified in attack_config
         batch_size=proxy_loader_batch_size,
         data_dir=data_dir,
         augment_train=False
     )

    # --- Load Pre-trained Victim Model ---
    print(f"\nLoading victim model structure using config: {args.victim_config}")
    print(f"Loading victim model weights from: {args.victim_checkpoint}")

    # Initialize victim model structure based on *its* config
    decoder_conf_victim = victim_config['victim_model']['decoder'].copy()
    if victim_config['task'] == 'classification':
        decoder_conf_victim['num_classes'] = num_classes

    victim_model = SC_Model(
        encoder_config=victim_config['victim_model']['encoder'],
        channel_config=victim_config['channel'], # Use victim's channel during attack eval if needed? Usually no.
        decoder_config=decoder_conf_victim,
        task=victim_config['task']
    ).to(device)

    # Load the saved weights
    try:
        victim_model.load_state_dict(torch.load(args.victim_checkpoint, map_location=device))
        victim_model.eval() # Set to evaluation mode
        print("Victim model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Victim checkpoint file not found at {args.victim_checkpoint}")
        return
    except Exception as e:
         print(f"Error loading victim checkpoint: {e}")
         return

    # --- Initialize Attacker Components ---
    print("\nInitializing attacker...")
    victim_interface = VictimQueryInterface(victim_model, attack_config['attacker'])

    # Ensure surrogate config has necessary info derived from victim/task
    surrogate_conf = attack_config['attacker']['surrogate_model'].copy()
    surrogate_conf['task'] = victim_config['task'] # Inherit task from victim
    if victim_config['task'] == 'classification':
        surrogate_conf['num_classes'] = num_classes

    attacker = ModelStealingAttacker(
        attack_config=attack_config['attacker'],
        victim_interface=victim_interface,
        surrogate_model_config=surrogate_conf,
        attacker_train_config=attack_config['attacker']['training_attacker'],
        proxy_dataloader=proxy_loader, # Give attacker the proxy loader
        device=device
    )

    # --- Run Attack ---
    attacker.run_attack() # This performs querying and surrogate training

    # --- Save Surrogate Model ---
    surrogate_save_path = os.path.join(args.output_dir, 'surrogate_model.pth')
    print(f"\nSaving trained surrogate model to: {surrogate_save_path}")
    torch.save(attacker.surrogate_model.state_dict(), surrogate_save_path)

    # --- Evaluate Attack ---
    # Note: Victim model's original test performance might be useful context,
    # but we won't re-evaluate it here unless needed. Assume it's known from its training run.
    attack_eval_results = attacker.evaluate_attack(test_loader) # Evaluate fidelity and surrogate task performance

    # --- Save Final Results ---
    results_save_path = os.path.join(args.output_dir, 'attack_results.json')
    print(f"\nSaving attack evaluation results to: {results_save_path}")
    # Convert tensors to lists/floats if necessary
    for key, value in attack_eval_results.items():
         if isinstance(value, torch.Tensor):
             attack_eval_results[key] = value.item()

    # Also save key attack config parameters with results
    final_results = {
        'attack_config_summary': {
            'type': attack_config['attacker']['type'],
            'query_access': attack_config['attacker']['query_access'],
            'latent_access': attack_config['attacker']['latent_access'],
            'query_budget': attack_config['attacker']['query_budget'],
            'final_query_count': victim_interface.get_query_count()
        },
        'evaluation_metrics': attack_eval_results
    }

    with open(results_save_path, 'w') as f:
        json.dump(final_results, f, indent=4)

    print("\nAttack simulation finished.")


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