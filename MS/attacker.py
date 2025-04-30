# attacker/attacker.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import random
import os
from tqdm import tqdm
import copy # To potentially deep copy model architectures

# Assuming these modules exist
from MS.query_interface import VictimQueryInterface
from models.resnet_sc import ResNetEncoderSC, ResNetDecoderSC # Can reuse victim architectures
# Or define specific surrogate models: from attacker.surrogate_model import SurrogateEncoder, SurrogateDecoder
from MS.train_surrogate import train_surrogate_epoch, evaluate_surrogate_fidelity # Need to define these
from evaluation.metrics import calculate_model_agreement, calculate_accuracy, calculate_psnr # Import necessary eval metrics

# --- Placeholder for Query Strategies ---
# You might define more sophisticated strategies later
class RandomQueryStrategy:
    """Selects random samples from a proxy dataset."""
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
             print(f"Warning: Proxy loader returned batch smaller than requested ({data.size(0)} < {batch_size}).")
             # Handle potentially smaller last batch, or refill? For simplicity, use what we got.
             pass # Use the smaller batch


        return data.to(device)

class ModelStealingAttacker:
    """
    Orchestrates the model stealing attack against an SC system.
    """
    def __init__(self,
                 attack_config: dict,
                 victim_interface: VictimQueryInterface,
                 surrogate_model_config: dict,
                 attacker_train_config: dict,
                 proxy_dataloader: DataLoader, # Used for generating initial queries
                 device: torch.device):
        """
        Initializes the attacker.

        Args:
            attack_config (dict): Configuration for the attack type, access, budget.
            victim_interface (VictimQueryInterface): Interface to query the victim.
            surrogate_model_config (dict): Configuration for the surrogate model architecture.
                                          Needs 'arch_name', 'latent_dim'.
                                          For classification decoder, needs 'num_classes'.
            attacker_train_config (dict): Configuration for training the surrogate.
                                           Needs 'lr', 'batch_size', 'epochs' or connects to budget.
            proxy_dataloader (DataLoader): DataLoader for a public or auxiliary dataset
                                           to generate queries from (attacker doesn't have victim's train data).
            device (torch.device): The device to run computations on.
        """
        self.attack_config = attack_config
        self.victim_interface = victim_interface
        self.surrogate_model_config = surrogate_model_config
        self.attacker_train_config = attacker_train_config
        self.proxy_dataloader = proxy_dataloader
        self.device = device

        self.attack_type = self.attack_config.get('type', 'steal_end2end') # steal_encoder, steal_decoder, steal_end2end
        self.query_access = self.attack_config.get('query_access', 'end_to_end_query')
        self.latent_access = self.attack_config.get('latent_access', 'none')
        self.query_budget = self.attack_config.get('query_budget', 10000) # Default budget
        self.collected_data = [] # List to store (query_input, victim_output, [optional_latent]) tuples

        # --- Initialize Surrogate Model ---
        self._initialize_surrogate_model()

        # --- Initialize Query Strategy ---
        # For now, use a simple random strategy from the proxy loader
        self.query_strategy = RandomQueryStrategy(self.proxy_dataloader)

        # --- Setup Training Components for Surrogate ---
        self.surrogate_optimizer = optim.Adam(
            self.surrogate_model.parameters(),
            lr=self.attacker_train_config.get('lr', 1e-3)
        )
        # Surrogate loss depends on what we are stealing
        self._setup_surrogate_loss()

        print(f"Initialized ModelStealingAttacker:")
        print(f"  Attack Type: {self.attack_type}")
        print(f"  Query Budget: {self.query_budget}")
        print(f"  Surrogate Model: {self.surrogate_model.__class__.__name__}")


    def _initialize_surrogate_model(self):
        """Initializes the surrogate model based on the attack type."""
        arch = self.surrogate_model_config['arch_name']
        latent_dim = self.surrogate_model_config['latent_dim']

        if self.attack_type == 'steal_encoder':
            # Attacker wants to replicate the encoder
            self.surrogate_model = ResNetEncoderSC(
                arch_name=arch,
                latent_dim=latent_dim,
                pretrained=False # Attacker trains from scratch
            ).to(self.device)
        elif self.attack_type == 'steal_decoder':
             # Attacker wants to replicate the decoder
             task = self.surrogate_model_config.get('task', 'reconstruction') # Need task info for decoder
             if task == 'reconstruction':
                 self.surrogate_model = ResNetDecoderSC(
                     arch_name=arch, # Should match victim's encoder arch
                     latent_dim=latent_dim,
                     output_channels=self.surrogate_model_config.get('output_channels', 3)
                 ).to(self.device)
             elif task == 'classification':
                  num_classes = self.surrogate_model_config.get('num_classes')
                  if num_classes is None: raise ValueError("Need num_classes for surrogate classification decoder")
                  # Simple MLP surrogate decoder/classifier head
                  self.surrogate_model = nn.Sequential(
                       nn.Linear(latent_dim, 512),
                       nn.ReLU(inplace=True),
                       nn.Dropout(p=self.surrogate_model_config.get('dropout', 0.5)),
                       nn.Linear(512, num_classes)
                  ).to(self.device)
             else:
                  raise ValueError(f"Unknown task for surrogate decoder: {task}")

        elif self.attack_type == 'steal_end2end':
             # Attacker builds a *direct* model from X to Y (doesn't necessarily have SC structure)
             # For simplicity, let's assume attacker uses a standard ResNet for classification
             # or an Autoencoder-like structure for reconstruction.
             # Or, attacker could try to replicate the *victim's* SC structure. Let's do that for consistency.
             print("Initializing end-to-end surrogate with SC structure (Encoder -> Decoder)")
             # NOTE: No channel simulated in the surrogate typically, unless attacker aims to steal robustness too.
             surrogate_encoder = ResNetEncoderSC(arch_name=arch, latent_dim=latent_dim, pretrained=False)
             task = self.surrogate_model_config.get('task', 'reconstruction')
             if task == 'reconstruction':
                  surrogate_decoder = ResNetDecoderSC(arch_name=arch, latent_dim=latent_dim, output_channels=self.surrogate_model_config.get('output_channels', 3))
             elif task == 'classification':
                  num_classes = self.surrogate_model_config.get('num_classes')
                  if num_classes is None: raise ValueError("Need num_classes for surrogate classification decoder")
                  surrogate_decoder = nn.Sequential(nn.Linear(latent_dim, num_classes)) # Simplest head
             else:
                  raise ValueError(f"Unknown task for surrogate end-to-end: {task}")

             # Combine them - This surrogate *doesn't* simulate the channel internally
             class SurrogateEnd2End(nn.Module):
                 def __init__(self, enc, dec):
                     super().__init__()
                     self.encoder = enc
                     self.decoder = dec
                 def forward(self, x):
                     z = self.encoder(x)
                     y = self.decoder(z)
                     return y
             self.surrogate_model = SurrogateEnd2End(surrogate_encoder, surrogate_decoder).to(self.device)

        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")

    def _setup_surrogate_loss(self):
        """Sets up the loss function for training the surrogate."""
        if self.attack_type == 'steal_encoder':
            # Match the latent representation z
            self.surrogate_criterion = nn.MSELoss() # Or L1Loss, Cosine Similarity Loss
            print("Using MSE loss for stealing encoder (matching latent z).")
        elif self.attack_type == 'steal_decoder':
            # Match the final output Y (reconstruction or classification)
            task = self.surrogate_model_config.get('task', 'reconstruction')
            if task == 'reconstruction':
                self.surrogate_criterion = nn.MSELoss() # Match reconstructed images
                print("Using MSE loss for stealing reconstruction decoder.")
            elif task == 'classification':
                 # Match the *logits* or use KL divergence for probabilities
                 # Using CrossEntropy assumes victim outputs labels, which it doesn't via query.
                 # MSE on logits is a common approach. KL divergence is better if victim provided probabilities.
                 self.surrogate_criterion = nn.MSELoss() # Match output logits
                 # self.surrogate_criterion = nn.KLDivLoss(reduction='batchmean') # Requires log_softmax on surrogate, softmax on victim
                 print("Using MSE loss for stealing classification decoder (matching logits).")
        elif self.attack_type == 'steal_end2end':
            # Match the final output Y
            task = self.surrogate_model_config.get('task', 'reconstruction')
            if task == 'reconstruction':
                 self.surrogate_criterion = nn.MSELoss()
                 print("Using MSE loss for stealing end-to-end reconstruction.")
            elif task == 'classification':
                 self.surrogate_criterion = nn.MSELoss() # Match logits
                 print("Using MSE loss for stealing end-to-end classification (matching logits).")
        else:
             raise ValueError(f"Cannot set loss for attack type: {self.attack_type}")


    def collect_data_batch(self, batch_size: int):
        """Queries the victim for one batch of data using the current strategy."""
        remaining_budget = self.query_budget - self.victim_interface.get_query_count()
        if remaining_budget <= 0:
            return False # Budget exhausted

        effective_batch_size = min(batch_size, remaining_budget)

        # 1. Get query inputs based on access type
        if self.query_access == 'encoder_query' or self.query_access == 'end_to_end_query':
             query_input = self.query_strategy.get_queries(effective_batch_size, self.device)
             if query_input.size(0) < effective_batch_size: # Handle smaller last batch from strategy
                 effective_batch_size = query_input.size(0)
                 if effective_batch_size == 0: return False # Strategy exhausted?

        elif self.query_access == 'decoder_query':
             # Attacker needs to generate z' inputs. How?
             # Option 1: Randomly sample from a prior distribution (e.g., Gaussian)
             # Option 2: Use the surrogate encoder (if available/trained) on proxy data
             # Let's use Option 1 for simplicity now.
             latent_dim = self.surrogate_model_config['latent_dim']
             query_input = torch.randn(effective_batch_size, latent_dim, device=self.device)
             # TODO: Implement more sophisticated z' generation if needed
        else:
             raise ValueError("Invalid query access for data collection")

        # 2. Query the victim
        victim_output_package = self.victim_interface.query(query_input)

        if victim_output_package is None: # Budget might have been exactly hit
            return False

        # 3. Store the data pair(s) - Adjust based on return format of query interface
        if self.latent_access == 'none':
            # Store (query_input, primary_output)
            self.collected_data.append((query_input.cpu(), victim_output_package.cpu()))
        else: # Grey-box returns tuple (primary_output, latent)
            primary_output, latent = victim_output_package
            self.collected_data.append((query_input.cpu(), primary_output.cpu(), latent.cpu()))

        return True # Data collected successfully

    def create_surrogate_dataset(self) -> Dataset:
        """Creates a PyTorch Dataset from the collected data."""
        if not self.collected_data:
            return None

        # Unzip the collected data
        # Note: The structure depends on the attack type and latent access
        inputs = []
        targets = []

        if self.attack_type == 'steal_encoder':
            # Input: X (from query_input), Target: Z_victim (from latent access or primary output)
            if self.latent_access == 'clean_z' or self.query_access == 'encoder_query':
                for x, _, z_v in self.collected_data: inputs.append(x); targets.append(z_v)
                # Special case: encoder_query, latent_access=none. Primary output is z.
                if self.latent_access == 'none' and self.query_access == 'encoder_query':
                     for x, z_v in self.collected_data: inputs.append(x); targets.append(z_v)
            else: raise ValueError("Cannot steal encoder without clean_z access or direct encoder query.")

        elif self.attack_type == 'steal_decoder':
             # Input: Z' (from query_input or latent access), Target: Y_victim (primary output)
             if self.query_access == 'decoder_query': # Input was z' (or z)
                 if self.latent_access == 'none':
                     for z_prime_in, y_v in self.collected_data: inputs.append(z_prime_in); targets.append(y_v)
                 elif self.latent_access == 'dirty_z_prime': # Input was z', also returned as latent
                     for z_prime_in, y_v, z_prime_latent in self.collected_data: inputs.append(z_prime_latent); targets.append(y_v)
                 else: # Clean Z access with decoder query doesn't make sense for training
                      raise ValueError("Cannot train steal_decoder surrogate with clean_z latent access via decoder_query.")
             elif self.query_access == 'end_to_end_query' and self.latent_access == 'dirty_z_prime':
                  # Input X, got Y and Z'. Train surrogate on Z' -> Y
                  for x, y_v, z_prime_latent in self.collected_data: inputs.append(z_prime_latent); targets.append(y_v)
             else: raise ValueError("Cannot steal decoder without decoder_query or end_to_end + dirty_z access.")

        elif self.attack_type == 'steal_end2end':
             # Input: X (from query_input), Target: Y_victim (primary output)
             if self.latent_access == 'none':
                 for x, y_v in self.collected_data: inputs.append(x); targets.append(y_v)
             else: # Grey-box also returns Y
                  for x, y_v, _ in self.collected_data: inputs.append(x); targets.append(y_v)
        else:
             raise ValueError(f"Unknown attack type: {self.attack_type}")


        if not inputs:
             print("Warning: No suitable data collected for training the surrogate based on config.")
             return None

        inputs_tensor = torch.cat(inputs, dim=0)
        targets_tensor = torch.cat(targets, dim=0)
        return TensorDataset(inputs_tensor, targets_tensor)


    def train_surrogate(self, epochs: int, batch_size: int):
        """Trains the surrogate model using the collected data."""
        print("\n--- Training Surrogate Model ---")
        surrogate_dataset = self.create_surrogate_dataset()
        if surrogate_dataset is None:
            print("No data available to train surrogate. Skipping.")
            return

        surrogate_loader = DataLoader(surrogate_dataset, batch_size=batch_size, shuffle=True)
        print(f"Created surrogate dataset with {len(surrogate_dataset)} samples.")

        for epoch in range(1, epochs + 1):
             epoch_start_time = time.time()
             avg_loss = train_surrogate_epoch(
                 self.surrogate_model,
                 surrogate_loader,
                 self.surrogate_optimizer,
                 self.surrogate_criterion,
                 self.device,
                 self.attack_type # Pass attack type if loss needs it
             )
             epoch_duration = time.time() - epoch_start_time
             print(f"Surrogate Train Epoch {epoch}/{epochs} | Time: {epoch_duration:.2f}s | Avg Loss: {avg_loss:.4f}")
             # TODO: Add evaluation on a hold-out set of collected data if desired

        print("--- Surrogate Training Finished ---")

    def run_attack(self):
        """Executes the full attack sequence: collect data and train surrogate."""
        print("\n--- Running Model Stealing Attack ---")
        queries_per_step = self.attacker_train_config.get('query_batch_size', 128)
        max_epochs = self.attacker_train_config.get('epochs', 10) # Can be fixed or adaptive

        # --- Data Collection Phase ---
        print("Starting data collection...")
        pbar = tqdm(total=self.query_budget, desc="Queries")
        initial_count = self.victim_interface.get_query_count()
        while self.victim_interface.get_query_count() < self.query_budget:
             success = self.collect_data_batch(queries_per_step)
             current_count = self.victim_interface.get_query_count()
             pbar.update(current_count - initial_count)
             initial_count = current_count
             if not success:
                 print("Data collection stopped (budget reached or error).")
                 break
        pbar.close()
        print(f"Data collection finished. Total queries made: {self.victim_interface.get_query_count()}")
        print(f"Collected {len(self.collected_data)} batches of data.")

        # --- Training Phase ---
        if self.collected_data:
             self.train_surrogate(
                 epochs=max_epochs,
                 batch_size=self.attacker_train_config.get('batch_size', 128)
             )
        else:
             print("No data collected, cannot train surrogate.")

        print("--- Attack Finished ---")

    def evaluate_attack(self, test_loader: DataLoader) -> dict:
        """
        Evaluates the success of the attack.

        Args:
            test_loader (DataLoader): The *original* test dataset loader
                                     (used to evaluate task performance).

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        print("\n--- Evaluating Attack Success ---")
        results = {}
        task = self.surrogate_model_config.get('task', 'reconstruction') # Get task from surrogate config

        # 1. Evaluate Surrogate Fidelity (Model Agreement or Output Similarity)
        print("Evaluating surrogate fidelity...")
        fidelity_metrics = evaluate_surrogate_fidelity(
            self.victim_interface, # Needs access to victim for comparison
            self.surrogate_model,
            test_loader, # Use original test data as input
            self.device,
            task,
            self.attack_type, # Need this to know how to query victim/surrogate
            self.latent_access # Need this potentially for grey-box comparison
        )
        results.update(fidelity_metrics)

        # 2. Evaluate Surrogate Task Performance
        # Run surrogate model on the original test set and calculate task metrics
        print("Evaluating surrogate task performance...")
        self.surrogate_model.eval()
        surrogate_task_eval = {}
        if self.attack_type == 'steal_encoder':
             print("Task performance evaluation not applicable for 'steal_encoder' attack.")
             pass # Cannot evaluate task performance with just an encoder
        elif self.attack_type == 'steal_decoder':
             # Need a way to get representative z' for the test set.
             # Option 1: Use victim encoder on test set + channel -> feed to surrogate decoder
             # Option 2: Sample random z' (less realistic)
             # Let's try Option 1 (requires victim encoder access conceptually)
             print("Evaluating surrogate decoder task performance (requires victim encoder + channel)...")
             all_targets = []
             all_surrogate_outputs = []
             with torch.no_grad():
                 for data, target in tqdm(test_loader, desc="Surrogate Decoder Eval", leave=False):
                     data = data.to(self.device)
                     z = self.victim_interface.victim_model.encode(data) # Get clean z
                     self.victim_interface.victim_model.channel.eval() # Ensure channel is eval
                     z_prime = self.victim_interface.victim_model.channel(z) # Simulate channel
                     y_surrogate = self.surrogate_model(z_prime) # Use surrogate decoder
                     all_targets.append(target.cpu())
                     all_surrogate_outputs.append(y_surrogate.cpu())

             all_targets = torch.cat(all_targets)
             all_surrogate_outputs = torch.cat(all_surrogate_outputs)

             if task == 'classification':
                  acc, _ = calculate_accuracy(all_surrogate_outputs, all_targets)
                  surrogate_task_eval['surrogate_task_accuracy'] = acc
             elif task == 'reconstruction':
                  # Need original images for comparison
                  all_original_data = []
                  for data, _ in test_loader: all_original_data.append(data)
                  all_original_data = torch.cat(all_original_data)
                  psnr = calculate_psnr(all_surrogate_outputs, all_original_data, data_range=2.0)
                  surrogate_task_eval['surrogate_task_psnr'] = psnr
                  # Add SSIM/LPIPS here if needed

        elif self.attack_type == 'steal_end2end':
             # Directly evaluate surrogate on test set
             all_targets = []
             all_surrogate_outputs = []
             with torch.no_grad():
                 for data, target in tqdm(test_loader, desc="Surrogate E2E Eval", leave=False):
                      data = data.to(self.device)
                      y_surrogate = self.surrogate_model(data)
                      all_targets.append(target.cpu())
                      all_surrogate_outputs.append(y_surrogate.cpu())
             all_targets = torch.cat(all_targets)
             all_surrogate_outputs = torch.cat(all_surrogate_outputs)

             if task == 'classification':
                  acc, _ = calculate_accuracy(all_surrogate_outputs, all_targets)
                  surrogate_task_eval['surrogate_task_accuracy'] = acc
             elif task == 'reconstruction':
                  all_original_data = []
                  for data, _ in test_loader: all_original_data.append(data)
                  all_original_data = torch.cat(all_original_data)
                  psnr = calculate_psnr(all_surrogate_outputs, all_original_data, data_range=2.0)
                  surrogate_task_eval['surrogate_task_psnr'] = psnr
                  # Add SSIM/LPIPS here if needed

        results.update(surrogate_task_eval)
        print(f"Attack Evaluation Results: {results}")
        return results


# NOTE: Need to implement train_surrogate_epoch and evaluate_surrogate_fidelity
# in training/train_attacker.py and evaluation/metrics.py (or keep fidelity eval here).
# Let's put helper train/eval for surrogate in a separate file for clarity.