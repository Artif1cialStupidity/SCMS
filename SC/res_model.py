# semantic_communication/model.py

import torch
import torch.nn as nn
from typing import Union, Tuple
from models.resnet_sc import ResNetEncoderSC, ResNetDecoderSC # Import models from models/
from SC.channel import get_channel # Import channel factory from channel.py

class SC_Model(nn.Module):
    """
    Semantic Communication System Model.
    Combines Encoder, Channel, and Decoder.
    """
    def __init__(self, encoder_config: dict, channel_config: dict, decoder_config: dict, task: str):
        """
        Initializes the end-to-end Semantic Communication model.

        Args:
            encoder_config (dict): Configuration for the encoder. Must include:
                                    'arch_name' (e.g., 'resnet18'),
                                    'latent_dim' (int),
                                    'pretrained' (bool, optional).
            channel_config (dict): Configuration for the channel. Must include:
                                    'type' (e.g., 'ideal', 'awgn', 'rayleigh'),
                                    'snr_db' (float or None, optional based on type).
                                    Other channel-specific params like 'rayleigh_add_awgn'.
            decoder_config (dict): Configuration for the decoder. Must include:
                                    'arch_name' (corresponding encoder arch),
                                    'latent_dim' (must match encoder's latent_dim),
                                    'output_channels' (int, optional, default 3 for image).
                                    Potentially params for classification head if task='classification'.
            task (str): The downstream task ('reconstruction' or 'classification').
                        This influences the decoder's final layer(s).
        """
        super().__init__()
        self.task = task.lower()
        if self.task not in ['reconstruction', 'classification']:
            raise ValueError(f"Unsupported task: {self.task}. Choose 'reconstruction' or 'classification'.")

        latent_dim = encoder_config['latent_dim']
        if latent_dim != decoder_config['latent_dim']:
             raise ValueError("Encoder and Decoder latent_dim must match!")

        # --- Initialize Components ---
        # 1. Encoder
        self.encoder = ResNetEncoderSC(
            arch_name=encoder_config['arch_name'],
            latent_dim=latent_dim,
            pretrained=encoder_config.get('pretrained', True) # Default to pretrained
        )

        # 2. Channel
        self.channel = get_channel(channel_config)

        # 3. Decoder
        if self.task == 'reconstruction':
            self.decoder = ResNetDecoderSC(
                arch_name=decoder_config['arch_name'],
                latent_dim=latent_dim,
                output_channels=decoder_config.get('output_channels', 3)
            )
        elif self.task == 'classification':
            # For classification, the 'decoder' might be simpler.
            # It takes z' and outputs class logits.
            # We can adapt the ResNetDecoderSC or create a specific ClassifierDecoder.
            # Example: Use a simple MLP head on top of z'
            num_classes = decoder_config.get('num_classes')
            if num_classes is None:
                raise ValueError("Decoder config must specify 'num_classes' for classification task.")

            # Simple MLP Decoder for Classification
            # We might not need the full ResNetDecoderSC structure here.
            # Let's define a simple MLP as the decoder/classifier head.
            # You could also potentially use parts of the ResNetDecoder's FC layer
            # followed by a classification layer if z' retains some spatial structure,
            # but if z' is just a flat vector from AWGN, an MLP is common.
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 512), # Example intermediate layer
                nn.ReLU(inplace=True),
                nn.Dropout(p=decoder_config.get('dropout', 0.5)), # Optional dropout
                nn.Linear(512, num_classes)
            )
            print(f"Initialized MLP Decoder for Classification (num_classes={num_classes})")

    def forward(self, x: torch.Tensor, return_latent: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Performs the end-to-end semantic communication process.

        Args:
            x (torch.Tensor): Input data tensor (e.g., image batch).
            return_latent (bool): If True, also returns the latent variables z and z_prime.
                                  Useful for specific loss calculations or analysis.

        Returns:
            torch.Tensor | tuple:
                If return_latent is False: The final output Y (reconstructed image or classification logits).
                If return_latent is True: A tuple (Y, z, z_prime).
        """
        # 1. Encode
        z = self.encoder(x)

        # --- Power Normalization (Optional but often recommended before channel) ---
        # Normalize z to have unit average power across the batch dimension
        # This makes the SNR definition in the channel more consistent.
        # Note: This adds complexity if latent_dim is very small.
        # mean_power = torch.mean(z**2, dim=1, keepdim=True)
        # z_normalized = z / torch.sqrt(mean_power + 1e-8) # Add epsilon for stability
        # --- End Power Normalization ---
        # If normalizing, use z_normalized below instead of z. Remember to potentially scale back at decoder if needed.
        # For simplicity now, we assume the channel handles power based on its init assumption.

        # 2. Transmit through Channel
        # Ensure channel is in the same training mode as the overall model
        self.channel.train(self.training)
        z_prime = self.channel(z) # Use z or z_normalized if implemented

        # 3. Decode
        y = self.decoder(z_prime)

        if return_latent:
            return y, z, z_prime
        else:
            return y

    # Optional: Method to directly query encoder (useful for attacker simulation)
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    # Optional: Method to directly query decoder (useful for attacker simulation)
    def decode(self, z_prime: torch.Tensor) -> torch.Tensor:
         # Ensure decoder gets input in the correct training mode if it has dropout etc.
         self.decoder.train(self.training)
         return self.decoder(z_prime)


# Example usage
if __name__ == '__main__':
    # Configuration Example
    enc_config = {'arch_name': 'resnet18', 'latent_dim': 64, 'pretrained': False}
    chan_config_awgn = {'type': 'awgn', 'snr_db': 10}
    chan_config_ideal = {'type': 'ideal'}

    # --- Reconstruction Task ---
    print("\n--- Testing Reconstruction Task ---")
    dec_config_recon = {'arch_name': 'resnet18', 'latent_dim': 64, 'output_channels': 3}
    sc_model_recon = SC_Model(enc_config, chan_config_awgn, dec_config_recon, task='reconstruction')
    sc_model_recon.train() # Set to training mode

    dummy_image = torch.randn(4, 3, 32, 32)
    output_recon, z_recon, z_prime_recon = sc_model_recon(dummy_image, return_latent=True)

    print(f"Input Image Shape: {dummy_image.shape}")
    print(f"Encoder Output (z) Shape: {z_recon.shape}")
    print(f"Channel Output (z') Shape: {z_prime_recon.shape}")
    print(f"Decoder Output (Recon Image) Shape: {output_recon.shape}")
    assert output_recon.shape == dummy_image.shape
    assert z_recon.shape == (4, enc_config['latent_dim'])
    assert z_prime_recon.shape == z_recon.shape

    # --- Classification Task ---
    print("\n--- Testing Classification Task ---")
    num_classes = 10 # Example for CIFAR-10
    dec_config_classify = {'arch_name': 'resnet18', 'latent_dim': 64, 'num_classes': num_classes, 'dropout': 0.5}
    # Using ideal channel for simplicity in this classification test
    sc_model_classify = SC_Model(enc_config, chan_config_ideal, dec_config_classify, task='classification')
    sc_model_classify.train()

    output_logits, z_classify, z_prime_classify = sc_model_classify(dummy_image, return_latent=True)

    print(f"Input Image Shape: {dummy_image.shape}")
    print(f"Encoder Output (z) Shape: {z_classify.shape}")
    print(f"Channel Output (z') Shape: {z_prime_classify.shape}") # z' == z for ideal channel
    print(f"Decoder Output (Logits) Shape: {output_logits.shape}")
    assert output_logits.shape == (4, num_classes)
    assert z_classify.shape == (4, enc_config['latent_dim'])
    assert torch.allclose(z_classify, z_prime_classify) # Check z' == z

    # Test encode/decode methods
    print("\n--- Testing encode/decode methods ---")
    z_direct = sc_model_classify.encode(dummy_image)
    assert torch.allclose(z_direct, z_classify)
    y_direct = sc_model_classify.decode(z_prime_classify)
    assert torch.allclose(y_direct, output_logits)
    print("encode/decode methods match forward pass.")