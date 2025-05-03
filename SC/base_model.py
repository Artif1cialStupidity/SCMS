# SC/base_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .channel import get_channel
from typing import Union, Tuple

# --- Helper: Residual Block ---
# Basic Residual Block (maintains dimensions)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Using kernel_size=3, padding=1, stride=1 maintains spatial dimensions
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True) # Apply ReLU after adding skip connection

    def forward(self, x):
        residual = x
        out = self.relu1(self.conv1(x))
        out = self.conv2(out)
        out += residual # Skip connection
        out = self.relu2(out) # Final ReLU after addition
        return out

# --- Transmitter (Encoder Component) ---
class BaseTransmitter(nn.Module):
    """Transmitter based on the provided table structure."""
    def __init__(self, input_channels=3, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # Input: (B, 3, 32, 32) assuming CIFAR-10/100

        # Layer Group 1: (Convolutional layer + ReLU) x 2 -> Output: 128 x 16 x 16
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        ) # Output: (B, 128, 16, 16)

        # Layer Group 2: Residual Block -> Output: 128 x 16 x 16
        self.res_block1 = ResidualBlock(128) # Output: (B, 128, 16, 16)

        # Layer Group 3: (Convolutional layer + ReLU) x 3 -> Output: 8 x 4 x 4
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 8x8 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 8, kernel_size=3, stride=1, padding=1),   # 4x4 -> 4x4, 128 -> 8 channels
            nn.ReLU(inplace=True)
        ) # Output: (B, 8, 4, 4)

        # Layer Group 4: Reshape + Dense + Tanh -> Output: 128 (latent_dim)
        self.final_block = nn.Sequential(
            nn.Flatten(), # Input: (B, 8*4*4 = 128)
            nn.Linear(8 * 4 * 4, latent_dim), # 128 -> 128
            nn.Tanh() # Output range [-1, 1]
        ) # Output: (B, 128)

        print(f"Initialized BaseTransmitter (latent_dim={latent_dim})")

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.res_block1(x)
        x = self.conv_block2(x)
        z = self.final_block(x)
        return z

# --- Receiver (Classifier Component) ---
class BaseReceiverClassifier(nn.Module):
    """Classifier Receiver based on the provided table structure."""
    def __init__(self, latent_dim=128, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.reshape_target_channels = 8
        self.reshape_target_spatial = 4

        # Layer 1: Dense + ReLU + Reshape -> Output: 8 x 4 x 4
        self.fc_reshape = nn.Sequential(
            nn.Linear(latent_dim, self.reshape_target_channels * self.reshape_target_spatial * self.reshape_target_spatial), # 128 -> 128
            nn.ReLU(inplace=True)
        ) # Output after reshape: (B, 8, 4, 4)

        # Layer 2: Convolutional layer + ReLU -> Output: 512 x 4 x 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.reshape_target_channels, 512, kernel_size=3, stride=1, padding=1), # 8 -> 512 channels
            nn.ReLU(inplace=True)
        ) # Output: (B, 512, 4, 4)

        # Layer 3: Residual Block -> Output: 512 x 4 x 4
        self.res_block1 = ResidualBlock(512) # Output: (B, 512, 4, 4)

        # Layer 4: Pooling -> Output: 512
        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # (B, 512, 1, 1)
            nn.Flatten()                  # (B, 512)
        ) # Output: (B, 512)

        # Layer 5: Dense + Softmax -> Output: 10 (Logits output)
        self.fc_final = nn.Linear(512, num_classes) # Output: (B, num_classes)

        print(f"Initialized BaseReceiverClassifier (num_classes={num_classes})")

    def forward(self, z_prime):
        x = self.fc_reshape(z_prime)
        x = x.view(x.size(0), self.reshape_target_channels, self.reshape_target_spatial, self.reshape_target_spatial) # Reshape
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.pooling(x)
        logits = self.fc_final(x)
        return logits


# --- Receiver (Decoder Component) ---
class BaseReceiverDecoder(nn.Module):
    """Decoder Receiver based on the provided table structure with Tanh output."""
    def __init__(self, latent_dim=128, output_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.reshape_target_channels = 8
        self.reshape_target_spatial = 4

        # Layer 1: Dense + Tanh + Reshape -> Output: 8 x 4 x 4
        self.fc_reshape = nn.Sequential(
            nn.Linear(latent_dim, self.reshape_target_channels * self.reshape_target_spatial * self.reshape_target_spatial), # 128 -> 128
            nn.Tanh() # Tanh activation here
        ) # Output after reshape: (B, 8, 4, 4)

        # Layer 2: Convolutional layer + ReLU -> Output: 512 x 4 x 4
        self.conv1 = nn.Sequential(
             nn.Conv2d(self.reshape_target_channels, 512, kernel_size=3, stride=1, padding=1), # 8 -> 512 channels
             nn.ReLU(inplace=True)
        ) # Output: (B, 512, 4, 4)

        # Layer 3: (Deconvolutional layer + ReLU) x 2 -> Output: 128 x 16 x 16
        self.deconv_block1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 4x4->8x8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 8x8->16x16
            nn.ReLU(inplace=True)
        ) # Output: (B, 128, 16, 16)

        # Layer 4: Residual Block -> Output: 128 x 16 x 16
        self.res_block1 = ResidualBlock(128) # Output: (B, 128, 16, 16)

        # Layer 5: Deconvolutional layer + ReLU -> Output: 64 x 32 x 32
        self.deconv2 = nn.Sequential(
             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 16x16->32x32
             nn.ReLU(inplace=True)
        ) # Output: (B, 64, 32, 32)

        # Layer 6: Convolutional layer + Tanh (CHANGED from Sigmoid) -> Output: 3 x 32 x 32
        self.conv_final = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1), # 64 -> 3 channels
            nn.Tanh() # ****** CHANGED to Tanh for [-1, 1] output range ******
        ) # Output: (B, 3, 32, 32)

        print(f"Initialized BaseReceiverDecoder (output_channels={output_channels}) with Tanh output.")

    def forward(self, z_prime):
        x = self.fc_reshape(z_prime)
        x = x.view(x.size(0), self.reshape_target_channels, self.reshape_target_spatial, self.reshape_target_spatial) # Reshape
        x = self.conv1(x)
        x = self.deconv_block1(x)
        x = self.res_block1(x)
        x = self.deconv2(x)
        img = self.conv_final(x)
        return img


# --- Encapsulating SC Model using Base Components ---
class Base_Model(nn.Module):
    """
    Semantic Communication System using the Base Transmitter/Receiver components.
    """
    def __init__(self, encoder_config: dict, channel_config: dict, decoder_config: dict, task: str):
        """
        Initializes the end-to-end SC model using Base components.

        Args:
            encoder_config (dict): Configuration for the encoder (BaseTransmitter). Expected keys:
                                    'latent_dim' (int, e.g., 128),
                                    'input_channels' (int, optional, default 3).
            channel_config (dict): Configuration for the channel (passed to get_channel).
            decoder_config (dict): Configuration for the decoder (BaseReceiver). Expected keys:
                                    'output_channels' (int, for reconstruction),
                                    'num_classes' (int, for classification).
                                    latent_dim is implicitly matched.
            task (str): The downstream task ('reconstruction' or 'classification').
        """
        super().__init__()
        self.task = task.lower()
        if self.task not in ['reconstruction', 'classification']:
            raise ValueError(f"Unsupported task: {self.task}. Choose 'reconstruction' or 'classification'.")

        # Get latent dim, use default from base model if not specified
        latent_dim = encoder_config.get('latent_dim', 128)
        if 'latent_dim' not in encoder_config: # Add for consistency if missing
            encoder_config['latent_dim'] = latent_dim

        # --- Initialize Components ---
        # 1. Encoder (Transmitter)
        self.encoder = BaseTransmitter(
            input_channels=encoder_config.get('input_channels', 3),
            latent_dim=latent_dim
        )

        # 2. Channel
        self.channel = get_channel(channel_config)

        # 3. Decoder (Receiver)
        if self.task == 'reconstruction':
            self.decoder = BaseReceiverDecoder(
                latent_dim=latent_dim, # Must match encoder
                output_channels=decoder_config.get('output_channels', 3)
            )
        elif self.task == 'classification':
            num_classes = decoder_config.get('num_classes')
            if num_classes is None:
                raise ValueError("Decoder config must specify 'num_classes' for classification task.")
            self.decoder = BaseReceiverClassifier(
                latent_dim=latent_dim, # Must match encoder
                num_classes=num_classes
            )

        print(f"Initialized Base_Model for task: {self.task}")

    def forward(self, x: torch.Tensor, return_latent: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Performs the end-to-end semantic communication process.

        Args:
            x (torch.Tensor): Input data tensor (e.g., image batch).
            return_latent (bool): If True, also returns z and z_prime.

        Returns:
            torch.Tensor | tuple: Final output Y or (Y, z, z_prime).
        """
        # 1. Encode
        z = self.encoder(x)

        # 2. Transmit through Channel
        self.channel.train(self.training) # Ensure channel mode matches model mode
        z_prime = self.channel(z)

        # 3. Decode
        y = self.decoder(z_prime)

        if return_latent:
            return y, z, z_prime
        else:
            return y

    # Optional: Method to directly query encoder
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    # Optional: Method to directly query decoder
    def decode(self, z_prime: torch.Tensor) -> torch.Tensor:
         self.decoder.train(self.training) # Ensure decoder mode matches
         return self.decoder(z_prime)


# --- Example Usage (for testing component shapes within the file) ---
if __name__ == '__main__':
    # Note: Running this directly might fail on 'from .channel import get_channel'
    # if not executed as part of the package structure.
    # These tests primarily verify the component shapes.

    latent_dim = 128
    num_classes = 10
    input_img = torch.randn(4, 3, 32, 32) # Batch of 4 CIFAR images

    print("\n--- Testing Transmitter Component ---")
    transmitter = BaseTransmitter(latent_dim=latent_dim)
    z = transmitter(input_img)
    print(f"Input shape: {input_img.shape}")
    print(f"Latent shape (z): {z.shape}")
    assert z.shape == (4, latent_dim)

    z_prime = z # Ideal channel

    print("\n--- Testing Classifier Receiver Component ---")
    classifier = BaseReceiverClassifier(latent_dim=latent_dim, num_classes=num_classes)
    logits = classifier(z_prime)
    print(f"Input shape (z'): {z_prime.shape}")
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (4, num_classes)

    print("\n--- Testing Decoder Receiver Component ---")
    decoder = BaseReceiverDecoder(latent_dim=latent_dim, output_channels=3)
    recon_img = decoder(z_prime)
    print(f"Input shape (z'): {z_prime.shape}")
    print(f"Reconstructed image shape: {recon_img.shape}")
    assert recon_img.shape == (4, 3, 32, 32)
    # Check output range is roughly [-1, 1] due to Tanh
    print(f"Decoder output min/max: {recon_img.min():.2f}/{recon_img.max():.2f}")


    # Test the full Base_Model encapsulation (requires dummy configs)
    print("\n--- Testing Base_Model Encapsulation ---")
    enc_conf = {'latent_dim': 128}
    chan_conf = {'type': 'ideal'} # Use ideal channel for simple test
    dec_conf_recon = {'output_channels': 3}
    dec_conf_classify = {'num_classes': 10}

    # Reconstruction Task
    base_model_recon = Base_Model(enc_conf, chan_conf, dec_conf_recon, task='reconstruction')
    y_recon, z_rec, zp_rec = base_model_recon(input_img, return_latent=True)
    print(f"Base Model Recon Output Shape: {y_recon.shape}")
    assert y_recon.shape == input_img.shape
    assert z_rec.shape == (4, latent_dim)

    # Classification Task
    base_model_classify = Base_Model(enc_conf, chan_conf, dec_conf_classify, task='classification')
    y_logits, z_cls, zp_cls = base_model_classify(input_img, return_latent=True)
    print(f"Base Model Classify Output Shape: {y_logits.shape}")
    assert y_logits.shape == (4, num_classes)
    assert z_cls.shape == (4, latent_dim)

    print("\nBase_Model shape tests passed.")