# attacker/query_interface.py

import torch
import torch.nn as nn
from semantic_communication.model import SC_Model # Import the victim model type
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
                                  - 'latent_access': 'none', 'clean_z', 'dirty_z_prime'
        """
        self.victim_model = victim_model
        self.victim_model.eval() # Ensure victim model is in evaluation mode

        self.query_access = attack_config.get('query_access', 'end_to_end_query').lower()
        self.latent_access = attack_config.get('latent_access', 'none').lower()
        self.query_count = 0
        self.query_budget = attack_config.get('query_budget', float('inf')) # Default to infinite budget

        # Validate configuration
        if self.query_access not in ['encoder_query', 'decoder_query', 'end_to_end_query']:
            raise ValueError(f"Invalid query_access type: {self.query_access}")
        if self.latent_access not in ['none', 'clean_z', 'dirty_z_prime']:
             raise ValueError(f"Invalid latent_access type: {self.latent_access}")

        # Grey-box access requires specific query types
        if self.latent_access == 'clean_z' and self.query_access == 'decoder_query':
             print("Warning: Requesting 'clean_z' access with 'decoder_query' is unusual. "
                   "'clean_z' is typically obtained via 'encoder_query' or 'end_to_end_query'.")
        if self.latent_access == 'dirty_z_prime' and self.query_access == 'encoder_query':
             print("Warning: Requesting 'dirty_z_prime' access with 'encoder_query' is unusual. "
                   "'dirty_z_prime' is typically obtained via 'end_to_end_query' or perhaps 'decoder_query' (though less direct).")

        print(f"Initialized VictimQueryInterface:")
        print(f"  Query Access Mode: {self.query_access}")
        print(f"  Latent Access Mode: {self.latent_access}")
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
                - Encoder query: Returns latent z.
                - Decoder query: Returns final output Y.
                - Grey-box: Returns the primary output plus the requested latent variable(s).
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

        with torch.no_grad(): # Queries should not compute gradients for the victim
            if self.query_access == 'encoder_query':
                # Attacker queries the encoder directly
                latent_z = self.victim_model.encode(query_input)
                primary_output = latent_z # Main output for this query type is z

                # Need to simulate z' if dirty_z access is requested (unusual case)
                if self.latent_access == 'dirty_z_prime':
                     # Pass the clean z through the victim's channel
                     self.victim_model.channel.eval() # Ensure channel is in eval mode if it behaves differently
                     latent_z_prime = self.victim_model.channel(latent_z)

            elif self.query_access == 'decoder_query':
                # Attacker provides z' (or z) and queries the decoder
                latent_input_to_decoder = query_input # Input IS z' (or z)
                primary_output = self.victim_model.decode(latent_input_to_decoder) # Main output is Y

                # Latent access in this mode is tricky.
                # 'dirty_z_prime' is the input itself.
                # 'clean_z' is not directly available unless the attacker *knew* the inverse channel.
                if self.latent_access == 'dirty_z_prime':
                     latent_z_prime = latent_input_to_decoder
                # Clean z access is not naturally provided here.

            elif self.query_access == 'end_to_end_query':
                 # Attacker provides X and gets Y (black-box) or Y + latent (grey-box)
                 # Use the forward pass that can return latent variables
                 y, z, z_p = self.victim_model(query_input, return_latent=True)
                 primary_output = y
                 latent_z = z
                 latent_z_prime = z_p

        # Increment query count (only if successful)
        self.query_count += query_input.size(0) # Increment by batch size

        # Prepare return value based on latent_access config
        if self.latent_access == 'none':
            return primary_output
        elif self.latent_access == 'clean_z':
            if latent_z is None:
                 print("Warning: 'clean_z' requested but not available for this query type.")
                 return primary_output # Return primary output only
            return primary_output, latent_z
        elif self.latent_access == 'dirty_z_prime':
             if latent_z_prime is None:
                 print("Warning: 'dirty_z_prime' requested but not available for this query type.")
                 return primary_output # Return primary output only
             return primary_output, latent_z_prime
        else:
             # Should not happen due to init validation
             return primary_output

    def get_query_count(self) -> int:
        """Returns the number of queries made so far."""
        return self.query_count

    def reset_query_count(self):
        """Resets the query counter."""
        self.query_count = 0


# Example Usage
if __name__ == '__main__':
    # Assume a trained SC_Model exists (we'll use a dummy one)
    from semantic_communication.model import SC_Model # Re-import locally for dummy creation
    enc_config = {'arch_name': 'resnet18', 'latent_dim': 64}
    chan_config = {'type': 'awgn', 'snr_db': 10}
    dec_config = {'arch_name': 'resnet18', 'latent_dim': 64, 'num_classes': 10} # Classification example
    dummy_victim = SC_Model(enc_config, chan_config, dec_config, task='classification')
    dummy_victim.eval() # Set to eval

    dummy_images = torch.randn(5, 3, 32, 32) # Batch of 5 images
    dummy_latents = torch.randn(5, 64) # Batch of 5 latents

    # --- Test Case 1: Black-box End-to-End ---
    print("\n--- Test: Black-box End-to-End ---")
    attack_config1 = {'query_access': 'end_to_end_query', 'latent_access': 'none', 'query_budget': 100}
    interface1 = VictimQueryInterface(dummy_victim, attack_config1)
    output_y1 = interface1.query(dummy_images)
    print(f"Query count: {interface1.get_query_count()}") # Should be 5
    print(f"Output Y shape: {output_y1.shape}") # Should be [5, 10] (logits)
    assert isinstance(output_y1, torch.Tensor) and output_y1.shape == (5, 10)

    # --- Test Case 2: Encoder Query ---
    print("\n--- Test: Encoder Query ---")
    attack_config2 = {'query_access': 'encoder_query', 'latent_access': 'none'} # No budget limit
    interface2 = VictimQueryInterface(dummy_victim, attack_config2)
    output_z2 = interface2.query(dummy_images)
    print(f"Query count: {interface2.get_query_count()}") # Should be 5
    print(f"Output Z shape: {output_z2.shape}") # Should be [5, 64]
    assert isinstance(output_z2, torch.Tensor) and output_z2.shape == (5, 64)

    # --- Test Case 3: Decoder Query ---
    print("\n--- Test: Decoder Query ---")
    attack_config3 = {'query_access': 'decoder_query', 'latent_access': 'none'}
    interface3 = VictimQueryInterface(dummy_victim, attack_config3)
    output_y3 = interface3.query(dummy_latents) # Input is z'
    print(f"Query count: {interface3.get_query_count()}") # Should be 5
    print(f"Output Y shape: {output_y3.shape}") # Should be [5, 10]
    assert isinstance(output_y3, torch.Tensor) and output_y3.shape == (5, 10)

    # --- Test Case 4: Grey-box End-to-End (Clean Z) ---
    print("\n--- Test: Grey-box End-to-End (Clean Z) ---")
    attack_config4 = {'query_access': 'end_to_end_query', 'latent_access': 'clean_z'}
    interface4 = VictimQueryInterface(dummy_victim, attack_config4)
    outputs4 = interface4.query(dummy_images)
    print(f"Query count: {interface4.get_query_count()}") # Should be 5
    assert isinstance(outputs4, tuple) and len(outputs4) == 2
    output_y4, output_z4 = outputs4
    print(f"Output Y shape: {output_y4.shape}") # Should be [5, 10]
    print(f"Output Z shape: {output_z4.shape}") # Should be [5, 64]
    assert output_y4.shape == (5, 10) and output_z4.shape == (5, 64)

    # --- Test Case 5: Grey-box End-to-End (Dirty Z') ---
    print("\n--- Test: Grey-box End-to-End (Dirty Z') ---")
    attack_config5 = {'query_access': 'end_to_end_query', 'latent_access': 'dirty_z_prime'}
    interface5 = VictimQueryInterface(dummy_victim, attack_config5)
    outputs5 = interface5.query(dummy_images)
    print(f"Query count: {interface5.get_query_count()}") # Should be 5
    assert isinstance(outputs5, tuple) and len(outputs5) == 2
    output_y5, output_z_prime5 = outputs5
    print(f"Output Y shape: {output_y5.shape}") # Should be [5, 10]
    print(f"Output Z' shape: {output_z_prime5.shape}") # Should be [5, 64]
    assert output_y5.shape == (5, 10) and output_z_prime5.shape == (5, 64)
    # Check if z' is different from z (due to noise)
    _, clean_z, _ = dummy_victim(dummy_images, return_latent=True)
    print(f"Norm difference between clean z and dirty z': {torch.norm(clean_z - output_z_prime5):.4f}") # Should be > 0

    # --- Test Case 6: Budget Exceeded ---
    print("\n--- Test: Budget Exceeded ---")
    attack_config6 = {'query_access': 'end_to_end_query', 'latent_access': 'none', 'query_budget': 3}
    interface6 = VictimQueryInterface(dummy_victim, attack_config6)
    output_y6 = interface6.query(dummy_images) # Uses 5 queries, exceeds budget
    print(f"Query count: {interface6.get_query_count()}") # Should be 5
    print(f"Output: {output_y6}") # Should be None
    assert output_y6 is None