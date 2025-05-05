# attacker/attacker.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import random
import os
import time # <-- Import time for training duration reporting
from tqdm import tqdm
import copy

from MS.query_interface import VictimQueryInterface
# Use the component import path directly
from models.components.resnet_components import ResNetEncoderSC # Or BaseTransmitter if needed
# Surrogate loss/eval helpers
from MS.train_surrogate import train_surrogate_epoch, evaluate_surrogate_fidelity
# Metrics specific for encoder comparison (can be in metrics.py or defined here)
from evaluation.metrics import calculate_psnr # Keep if needed elsewhere, but focus on latent metrics
import torch.nn.functional as F # For MSE Loss, Cosine Similarity

# --- RandomQueryStrategy remains the same ---
class RandomQueryStrategy:
    # ... (implementation as before) ...
    def __init__(self, proxy_dataloader: DataLoader):
        self.proxy_dataloader = proxy_dataloader
        self.iterator = iter(self.proxy_dataloader)

    def get_queries(self, batch_size: int, device: torch.device):
        """Gets the next batch of queries."""
        try:
            data, _ = next(self.iterator) # Ignore labels from proxy set if any
        except StopIteration:
            # Reset iterator if proxy dataset is exhausted
            self.iterator = iter(self.proxy_dataloader)
            data, _ = next(self.iterator)

        # Ensure correct batch size and move to device
        if data.size(0) > batch_size:
            data = data[:batch_size]
        elif data.size(0) < batch_size:
             # print(f"Warning: Proxy loader returned batch smaller than requested ({data.size(0)} < {batch_size}). Using smaller batch.")
             pass # Use the smaller batch

        return data.to(device)

class ModelStealingAttacker:
    """
    Orchestrates the model stealing attack against an SC system's ENCODER.
    """
    def __init__(self,
                 attack_config: dict,
                 victim_interface: VictimQueryInterface,
                 surrogate_model_config: dict, # Config for the SURROGATE ENCODER
                 attacker_train_config: dict,
                 proxy_dataloader: DataLoader,
                 device: torch.device):
        """
        Initializes the attacker for stealing the encoder.

        Args:
            attack_config (dict): Config for attack (type, access, budget, noise_scale).
            victim_interface (VictimQueryInterface): Interface to query the victim.
            surrogate_model_config (dict): Config for the surrogate ENCODER architecture
                                          (e.g., 'arch_name', 'latent_dim').
            attacker_train_config (dict): Config for training the surrogate (lr, batch_size, epochs).
            proxy_dataloader (DataLoader): DataLoader for generating queries.
            device (torch.device): Device for computations.
        """
        self.attack_config = attack_config
        self.victim_interface = victim_interface
        self.surrogate_model_config = surrogate_model_config
        self.attacker_train_config = attacker_train_config
        self.proxy_dataloader = proxy_dataloader
        self.device = device

        # --- Enforce Encoder Stealing Focus ---
        self.attack_type = self.attack_config.get('type')
        if self.attack_type != 'steal_encoder':
            raise ValueError(f"This attacker implementation currently only supports 'steal_encoder'. Found type: {self.attack_type}")

        self.query_access = self.attack_config.get('query_access')
        self.latent_access = self.attack_config.get('latent_access')
        # Ensure latent access is suitable for getting noisy z
        if self.latent_access not in ['noisy_scaled_z', 'clean_z']: # clean_z could be a baseline
             print(f"Warning: Latent access is {self.latent_access}. Ensure this provides the desired (X, z_observed) pairs for training the encoder.")
             if self.latent_access != 'noisy_scaled_z':
                  print("Ensure the 'noise_scale' parameter is handled appropriately or ignored if not needed.")


        self.query_budget = self.attack_config.get('query_budget', 10000)
        self.collected_data = [] # List to store (query_input_X, observed_latent_z) tuples

        # --- Initialize Surrogate ENCODER Model ---
        self._initialize_surrogate_encoder() # Renamed method

        # --- Initialize Query Strategy ---
        self.query_strategy = RandomQueryStrategy(self.proxy_dataloader)

        # --- Setup Training Components for Surrogate ENCODER ---
        self.surrogate_optimizer = optim.Adam(
            self.surrogate_encoder.parameters(), # Use surrogate_encoder
            lr=self.attacker_train_config.get('lr', 1e-3)
        )
        # Surrogate loss: Match the observed latent representation z
        self.surrogate_criterion = nn.MSELoss() # Or CosineEmbeddingLoss, L1Loss
        print("Using MSE loss for stealing encoder (matching observed latent z).")


        print(f"Initialized ModelStealingAttacker (Encoder Focus):")
        print(f"  Query Access: {self.query_access}")
        print(f"  Latent Access: {self.latent_access}")
        if self.latent_access == 'noisy_scaled_z':
            print(f"  Noise Scale: {self.attack_config.get('noise_scale')}")
        print(f"  Query Budget: {self.query_budget}")
        print(f"  Surrogate Encoder: {self.surrogate_encoder.__class__.__name__} ({self.surrogate_model_config['arch_name']})")


    def _initialize_surrogate_encoder(self):
        """Initializes the surrogate ENCODER model."""
        arch = self.surrogate_model_config['arch_name']
        latent_dim = self.surrogate_model_config['latent_dim']

        # Attacker wants to replicate the encoder X -> z
        # Use a suitable encoder architecture, e.g., ResNetEncoderSC
        # Or potentially BaseTransmitter if specified
        if 'resnet' in arch.lower():
            self.surrogate_encoder = ResNetEncoderSC(
                arch_name=arch,
                latent_dim=latent_dim,
                pretrained=False # Attacker trains from scratch
            ).to(self.device)
        # Add BaseTransmitter option if needed
        # elif arch == 'base_transmitter':
        #     self.surrogate_encoder = BaseTransmitter(...)
        else:
            # Fallback or error for unknown arch
             print(f"Warning: Unknown surrogate architecture '{arch}'. Using ResNetEncoderSC as default.")
             self.surrogate_encoder = ResNetEncoderSC(
                 arch_name='resnet18', # Default arch
                 latent_dim=latent_dim,
                 pretrained=False
            ).to(self.device)


    # Remove _setup_surrogate_loss method - logic moved to init

    def collect_data_batch(self, batch_size: int):
        """Queries the victim for one batch of data using the current strategy."""
        remaining_budget = self.query_budget - self.victim_interface.get_query_count()
        if remaining_budget <= 0:
            return False # Budget exhausted

        effective_batch_size = min(batch_size, remaining_budget)

        # 1. Get query inputs (X)
        # Requires query_access that takes X as input
        if self.query_access not in ['encoder_query', 'end_to_end_query']:
             raise ValueError(f"Cannot steal encoder with query_access '{self.query_access}'. Needs 'encoder_query' or 'end_to_end_query'.")

        query_input_x = self.query_strategy.get_queries(effective_batch_size, self.device)
        if query_input_x.size(0) == 0: return False # Strategy exhausted?
        effective_batch_size = query_input_x.size(0) # Adjust if last batch was smaller

        # 2. Query the victim
        victim_output_package = self.victim_interface.query(query_input_x)

        if victim_output_package is None: # Budget might have been exactly hit or error
            print("Warning: Victim query returned None. Budget might be hit or interface error.")
            return False

        # 3. Store the data pair(s) - We need (X, z_observed)
        # The interface now returns (primary_output, observed_latent) or just primary_output
        if isinstance(victim_output_package, tuple) and len(victim_output_package) == 2:
            _, observed_latent_z = victim_output_package # Discard primary_output, keep observed_latent
            if observed_latent_z is None:
                 print("Warning: Query returned tuple but observed_latent is None. Skipping batch.")
                 return True # Allow continuing collection, but this batch is lost
        elif self.latent_access == 'clean_z' and self.query_access == 'encoder_query':
             # Special case: encoder_query with clean_z access (no tuple returned by default)
             observed_latent_z = victim_output_package # The primary output *is* clean_z
        else:
            print(f"Warning: Unexpected victim output format or missing latent. Type: {type(victim_output_package)}. Skipping batch.")
            # Attempt to get query count to see if budget was the issue
            print(f"Current query count: {self.victim_interface.get_query_count()} / {self.query_budget}")
            return True # Allow continuing collection maybe? Or return False? Let's continue but log.


        # Store (query_input_X, observed_latent_z)
        self.collected_data.append((query_input_x.cpu(), observed_latent_z.cpu()))

        return True # Data collected successfully

    def create_surrogate_dataset(self) -> Dataset:
        """Creates a PyTorch Dataset from the collected (X, z_observed) data."""
        if not self.collected_data:
            print("No data collected for surrogate training.")
            return None

        # Unzip the collected data: list of (X_batch, z_observed_batch) tuples
        inputs_x = []
        targets_z = []
        for x, z_observed in self.collected_data:
            inputs_x.append(x)
            targets_z.append(z_observed)

        if not inputs_x:
             print("Warning: Failed to extract data for surrogate dataset.")
             return None

        inputs_tensor = torch.cat(inputs_x, dim=0)
        targets_tensor = torch.cat(targets_z, dim=0)
        print(f"Created surrogate dataset with {inputs_tensor.size(0)} samples.")
        return TensorDataset(inputs_tensor, targets_tensor)


    def train_surrogate(self, epochs: int, batch_size: int):
        """Trains the surrogate ENCODER model using the collected data."""
        print("\n--- Training Surrogate Encoder ---")
        surrogate_dataset = self.create_surrogate_dataset()
        if surrogate_dataset is None or len(surrogate_dataset) == 0:
            print("No data available to train surrogate encoder. Skipping.")
            return

        surrogate_loader = DataLoader(surrogate_dataset, batch_size=batch_size, shuffle=True)
        print(f"Training surrogate encoder on {len(surrogate_dataset)} samples for {epochs} epochs.")

        # Ensure the correct model is being trained
        self.surrogate_encoder.train()

        for epoch in range(1, epochs + 1):
             epoch_start_time = time.time()
             # Pass the surrogate encoder to the training function
             avg_loss = train_surrogate_epoch(
                 model=self.surrogate_encoder, # Pass the encoder
                 loader=surrogate_loader,
                 optimizer=self.surrogate_optimizer,
                 criterion=self.surrogate_criterion,
                 device=self.device,
                 attack_type=self.attack_type, # Still useful for potential logging/logic inside train_epoch
                 # task='latent_matching' # Can add more specific task info if needed
             )
             epoch_duration = time.time() - epoch_start_time
             print(f"Surrogate Encoder Train Epoch {epoch}/{epochs} | Time: {epoch_duration:.2f}s | Avg Loss: {avg_loss:.6f}")
             # TODO: Add evaluation on a hold-out set of collected data if desired

        print("--- Surrogate Encoder Training Finished ---")
        self.surrogate_encoder.eval() # Set back to eval mode after training

    def run_attack(self):
        """Executes the attack: collect data and train surrogate encoder."""
        print("\n--- Running Model Stealing Attack (Encoder Focus) ---")
        queries_per_step = self.attacker_train_config.get('query_batch_size', 128)
        max_epochs = self.attacker_train_config.get('epochs', 10)

        # --- Data Collection Phase ---
        print("Starting data collection...")
        pbar = tqdm(total=self.query_budget, desc="Queries")
        initial_count = self.victim_interface.get_query_count()
        collected_samples = 0
        while self.victim_interface.get_query_count() < self.query_budget:
             success = self.collect_data_batch(queries_per_step)
             current_count = self.victim_interface.get_query_count()
             delta_count = current_count - initial_count
             pbar.update(delta_count)
             if success and delta_count > 0 and self.collected_data:
                 # Estimate collected samples based on last batch size added
                 collected_samples += self.collected_data[-1][0].size(0)
             initial_count = current_count
             if not success and self.victim_interface.get_query_count() >= self.query_budget:
                 # Expected stop due to budget
                 break
             elif not success:
                  # Unexpected stop (e.g., query error, strategy exhausted)
                  print("\nData collection stopped potentially before budget reached.")
                  break
        pbar.close()
        print(f"Data collection finished. Total queries used: {self.victim_interface.get_query_count()} / {self.query_budget}")
        print(f"Collected {len(self.collected_data)} batches, approx {collected_samples} (X, z_observed) pairs.") # Adjusted log

        # --- Training Phase ---
        if self.collected_data:
             self.train_surrogate(
                 epochs=max_epochs,
                 batch_size=self.attacker_train_config.get('batch_size', 128)
             )
        else:
             print("No data collected, cannot train surrogate encoder.")

        print("--- Attack Finished ---")

    def evaluate_attack(self, test_loader: DataLoader) -> dict:
        """
        Evaluates the success of the attack by measuring surrogate encoder fidelity.

        Args:
            test_loader (DataLoader): The *original* test dataset loader.

        Returns:
            dict: Dictionary containing encoder fidelity metrics (e.g., latent_mse, latent_cosine_similarity).
        """
        print("\n--- Evaluating Attack Success (Encoder Fidelity) ---")
        results = {}

        # We only evaluate fidelity for steal_encoder
        print("Evaluating surrogate encoder fidelity vs victim encoder...")

        # Ensure surrogate is in eval mode
        self.surrogate_encoder.eval()

        # Pass the actual victim encoder and surrogate encoder to the evaluation function
        # The victim_interface holds the victim_model which has the encoder
        if not hasattr(self.victim_interface.victim_model, 'encoder'):
             print("Error: Victim model in interface does not have an 'encoder' attribute.")
             return {"error": "Victim encoder not accessible"}
        if not hasattr(self, 'surrogate_encoder'):
            print("Error: Attacker does not have a 'surrogate_encoder' attribute.")
            return {"error": "Surrogate encoder not found"}


        fidelity_metrics = evaluate_surrogate_fidelity(
            victim_encoder=self.victim_interface.victim_model.encoder, # Pass victim's actual encoder
            surrogate_encoder=self.surrogate_encoder,                  # Pass surrogate encoder
            test_loader=test_loader,
            device=self.device,
            # task=self.surrogate_model_config.get('task', 'reconstruction') # Task isn't directly relevant for z comparison
            # attack_type=self.attack_type # Implicitly steal_encoder
            # latent_access=self.latent_access # Not needed for final Z vs Z comparison
        )
        results.update(fidelity_metrics)

        # Task performance is not evaluated as we only stole the encoder
        print("Surrogate task performance evaluation is N/A for 'steal_encoder'.")

        print(f"Attack Evaluation Results (Encoder Fidelity): {results}")
        return results