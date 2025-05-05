# attacker/query_interface.py

import torch
import torch.nn as nn
# from SC.model import SC_Model # Changed in V2 - Already imported
from models.model import SC_Model # <-- Make sure using the unified model
from typing import Union, Tuple

class VictimQueryInterface:
    """
    Simulates the attacker's query access to the victim SC model,
    enforcing black-box or grey-box constraints.
    """
    def __init__(self, victim_model: SC_Model, attack_config: dict):
        """
        Initializes the query interface.

        Args:
            victim_model (SC_Model): The *trained* victim semantic communication model.
            attack_config (dict): Configuration dictionary for the attack, specifying
                                  access levels. Expected keys:
                                  - 'query_access': 'encoder_query', 'decoder_query', 'end_to_end_query'
                                  - 'latent_access': 'none', 'clean_z', 'dirty_z_prime', 'noisy_scaled_z' # <-- Added option
                                  - 'noise_scale': float (required if latent_access='noisy_scaled_z') # <-- Added config
        """
        self.victim_model = victim_model
        self.victim_model.eval() # Ensure victim model is in evaluation mode

        self.query_access = attack_config.get('query_access', 'end_to_end_query').lower()
        self.latent_access = attack_config.get('latent_access', 'none').lower()
        self.noise_scale = attack_config.get('noise_scale', 0.0) # <-- Get noise scale
        self.query_count = 0
        self.query_budget = attack_config.get('query_budget', float('inf')) # Default to infinite budget

        # Validate configuration
        if self.query_access not in ['encoder_query', 'decoder_query', 'end_to_end_query']:
            raise ValueError(f"Invalid query_access type: {self.query_access}")
        # --- Updated latent access validation ---
        valid_latent_access = ['none', 'clean_z', 'dirty_z_prime', 'noisy_scaled_z']
        if self.latent_access not in valid_latent_access:
             raise ValueError(f"Invalid latent_access type: {self.latent_access}. Valid options: {valid_latent_access}")
        # --- Validation for noise_scale ---
        if self.latent_access == 'noisy_scaled_z' and self.noise_scale is None:
            raise ValueError("`noise_scale` must be provided in attack_config when latent_access is 'noisy_scaled_z'")
        if self.latent_access == 'noisy_scaled_z' and (self.query_access == 'decoder_query'):
             print("Warning: Requesting 'noisy_scaled_z' access with 'decoder_query' is unusual. "
                   "This access requires calculating z and z' from an input X, which isn't provided in decoder_query.")


        # Grey-box access requires specific query types (Existing warnings remain valid)
        # ... (keep existing warnings for clean_z/dirty_z_prime combinations) ...

        print(f"Initialized VictimQueryInterface:")
        print(f"  Query Access Mode: {self.query_access}")
        print(f"  Latent Access Mode: {self.latent_access}")
        if self.latent_access == 'noisy_scaled_z':
             print(f"  Noise Scale for noisy_scaled_z: {self.noise_scale}") # <-- Print scale
        print(f"  Query Budget: {self.query_budget}")


    def query(self, query_input: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Performs a query to the victim model based on the allowed access.

        Args:
            query_input (torch.Tensor): The input provided by the attacker.
                                       - If query_access is 'encoder_query' or 'end_to_end_query',
                                         this should be a batch of input data X (e.g., images).
                                       - If query_access is 'decoder_query', this should be a
                                         batch of latent representations z' (or z if simulating).

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
                The output(s) allowed by the access configuration.
                - Black-box end-to-end: Returns only final output Y.
                - Encoder query: Returns latent z (unless latent_access modifies it).
                - Decoder query: Returns final output Y.
                - Grey-box: Returns the primary output plus the requested latent variable(s).
                - 'noisy_scaled_z': Returns primary_output, z + scale * (z' - z).
                Returns None if query budget is exceeded.
        """
        if self.query_count >= self.query_budget:
            print("Query budget exceeded.")
            return None

        # Ensure input is on the correct device
        device = next(self.victim_model.parameters()).device
        query_input = query_input.to(device)

        primary_output = None
        latent_z = None
        latent_z_prime = None
        observed_latent = None # Will hold the specific latent returned based on config

        with torch.no_grad(): # Queries should not compute gradients for the victim
            if self.query_access == 'encoder_query':
                # Attacker queries the encoder directly
                latent_z = self.victim_model.encode(query_input)
                primary_output = latent_z # Main output for this query type is z by default

                # --- Handle latent access variations for encoder_query ---
                if self.latent_access == 'none':
                    observed_latent = primary_output # Just return z
                elif self.latent_access == 'clean_z':
                    observed_latent = latent_z
                elif self.latent_access == 'dirty_z_prime':
                    # Need to simulate z'
                    self.victim_model.channel.eval()
                    latent_z_prime = self.victim_model.channel(latent_z)
                    observed_latent = latent_z_prime
                elif self.latent_access == 'noisy_scaled_z':
                    # Calculate z', noise, and scaled noisy z
                    self.victim_model.channel.eval()
                    latent_z_prime = self.victim_model.channel(latent_z)
                    channel_noise = latent_z_prime - latent_z
                    observed_latent = latent_z + channel_noise * self.noise_scale
                    # The primary output remains z, but the returned latent is modified
                # --- End latent access variations ---


            elif self.query_access == 'decoder_query':
                # Attacker provides z' (or z) and queries the decoder
                latent_input_to_decoder = query_input # Input IS z' (or z)
                primary_output = self.victim_model.decode(latent_input_to_decoder) # Main output is Y

                # Latent access in this mode
                if self.latent_access == 'none':
                    observed_latent = None # No latent access requested
                elif self.latent_access == 'dirty_z_prime':
                     observed_latent = latent_input_to_decoder # The input z' is the 'dirty' latent
                elif self.latent_access == 'clean_z':
                     # Cannot generally get clean_z from decoder query unless attacker knows inverse channel
                     observed_latent = None
                     print("Warning: 'clean_z' requested with 'decoder_query', but it's not available.")
                elif self.latent_access == 'noisy_scaled_z':
                     # Cannot calculate this without X -> z -> z' path
                     observed_latent = None
                     print("Warning: 'noisy_scaled_z' requested with 'decoder_query', but it cannot be calculated.")
                # Note: primary_output IS the main return value here


            elif self.query_access == 'end_to_end_query':
                 # Attacker provides X and gets Y (or Y + latent)
                 y, z, z_p = self.victim_model(query_input, return_latent=True)
                 primary_output = y
                 latent_z = z
                 latent_z_prime = z_p # Use the z_p calculated by the model

                 # --- Handle latent access variations for end_to_end_query ---
                 if self.latent_access == 'none':
                      observed_latent = None
                 elif self.latent_access == 'clean_z':
                      observed_latent = latent_z
                 elif self.latent_access == 'dirty_z_prime':
                      observed_latent = latent_z_prime
                 elif self.latent_access == 'noisy_scaled_z':
                      channel_noise = latent_z_prime - latent_z
                      observed_latent = latent_z + channel_noise * self.noise_scale
                 # --- End latent access variations ---

        # Increment query count
        self.query_count += query_input.size(0)

        # --- Updated Return Logic ---
        if self.latent_access == 'none':
            return primary_output
        elif observed_latent is not None:
             # Always return primary_output + the specific observed_latent determined by config
             return primary_output, observed_latent
        else:
             # Case where latent access was requested but couldn't be provided (e.g., noisy_scaled_z with decoder_query)
             # Or if latent_access was 'none' during decoder_query. Return only primary.
             return primary_output

    def get_query_count(self) -> int:
        """Returns the number of queries made so far."""
        return self.query_count

    def reset_query_count(self):
        """Resets the query counter."""
        self.query_count = 0

# --- Keep the __main__ block for testing if desired, update tests for noisy_scaled_z ---
# Example Test Case for noisy_scaled_z (add to __main__):
# if __name__ == '__main__':
#     ... (existing setup) ...
#     print("\n--- Test: End-to-End Query (Noisy Scaled Z) ---")
#     attack_config_noisy = {'query_access': 'end_to_end_query',
#                            'latent_access': 'noisy_scaled_z',
#                            'noise_scale': 0.5, # Example scale
#                            'query_budget': 100}
#     interface_noisy = VictimQueryInterface(dummy_victim, attack_config_noisy)
#     outputs_noisy = interface_noisy.query(dummy_images)
#     print(f"Query count: {interface_noisy.get_query_count()}")
#     assert isinstance(outputs_noisy, tuple) and len(outputs_noisy) == 2
#     output_y_noisy, output_z_observed_noisy = outputs_noisy
#     print(f"Output Y shape: {output_y_noisy.shape}")
#     print(f"Output Observed Z shape: {output_z_observed_noisy.shape}")
#     assert output_y_noisy.shape == (5, 10) and output_z_observed_noisy.shape == (5, 64)
#     # Verify z_observed is between z and z'
#     _, clean_z, dirty_z_prime = dummy_victim(dummy_images, return_latent=True)
#     expected_z_observed = clean_z + (dirty_z_prime - clean_z) * 0.5
#     print(f"Norm difference between calculated observed z and expected: {torch.norm(output_z_observed_noisy - expected_z_observed):.4f}") # Should be close to 0