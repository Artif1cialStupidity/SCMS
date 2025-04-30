# config_utils.py

import yaml
import os
import pprint
from typing import Union, Dict, Any

def load_config(config_path: str) -> Union[Dict[Any, Any], None]:
    """
    Loads configuration from a YAML file with error handling.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict | None: The loaded configuration dictionary, or None if an error occurs.
    """
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at '{config_path}'")
        return None
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from: {config_path}")
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file '{config_path}': {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading config '{config_path}': {e}")
        return None

def save_config(config: dict, save_path: str):
    """Saves a configuration dictionary to a YAML file."""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Configuration saved to: {save_path}")
    except Exception as e:
        print(f"Error saving configuration to '{save_path}': {e}")

def pretty_print_config(config: dict):
    """Prints the configuration dictionary in a readable format."""
    print("\n--- Configuration ---")
    pprint.pprint(config, indent=2)
    print("---------------------\n")

# Example Usage (Optional: for testing this file)
if __name__ == '__main__':
    # Create a dummy config file for testing
    dummy_cfg = {'dataset': 'cifar10', 'model': {'arch': 'resnet18'}}
    dummy_path = './dummy_config_test.yaml'
    save_config(dummy_cfg, dummy_path)

    # Test loading
    loaded_cfg = load_config(dummy_path)
    if loaded_cfg:
        pretty_print_config(loaded_cfg)

    # Test loading non-existent file
    load_config('./non_existent_config.yaml')

    # Clean up dummy file
    if os.path.exists(dummy_path):
        os.remove(dummy_path)