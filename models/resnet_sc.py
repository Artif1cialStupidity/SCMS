# models/resnet_sc.py

import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

# Helper function to get a pre-trained ResNet model without the final FC layer
def _get_resnet_backbone(arch_name: str, pretrained: bool = True):
    """Loads a pretrained ResNet model and removes the final classification layer."""
    if arch_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        feature_dim = 512 # ResNet18/34 output feature dimension
    elif arch_name == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        feature_dim = 512
    elif arch_name == 'resnet50':
         model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
         feature_dim = 2048 # ResNet50+ output feature dimension
    # Add more architectures if needed (resnet101, resnet152)
    else:
        raise ValueError(f"Unsupported ResNet architecture: {arch_name}")

    # Remove the final fully connected layer (classifier)
    modules = list(model.children())[:-1]
    backbone = nn.Sequential(*modules)
    return backbone, feature_dim

# --- Semantic Encoder ---
class ResNetEncoderSC(nn.Module):
    """ResNet-based Semantic Encoder."""
    def __init__(self, arch_name: str, latent_dim: int, pretrained: bool = True):
        """
        Args:
            arch_name (str): Name of the ResNet architecture (e.g., 'resnet18', 'resnet34').
            latent_dim (int): The desired dimension of the output latent representation 'z'.
            pretrained (bool): Whether to load pretrained weights for the ResNet backbone.
        """
        super().__init__()
        self.backbone, feature_dim = _get_resnet_backbone(arch_name, pretrained)
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # Ensure spatial dimensions are 1x1
            nn.Flatten(),                 # Flatten to a vector
            nn.Linear(feature_dim, latent_dim),
            # Optional: Add activation like ReLU or Tanh, or keep linear
            # nn.ReLU(inplace=True)
            # nn.Tanh()
        )
        print(f"Initialized ResNetEncoderSC (arch={arch_name}, latent_dim={latent_dim}, pretrained={pretrained})")
        print(f"  Backbone output features: {feature_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input image tensor into a latent semantic representation.

        Args:
            x (torch.Tensor): Input image tensor (batch_size, 3, H, W).
                              For CIFAR, H=W=32.

        Returns:
            torch.Tensor: Latent representation tensor 'z' (batch_size, latent_dim).
        """
        features = self.backbone(x)
        z = self.projection(features)
        return z

# --- Semantic Decoder ---
# Note: Building a good ResNet-style decoder is non-trivial.
# This is a simplified example using transposed convolutions.
# More sophisticated architectures might be needed for high-quality reconstruction.
class ResNetDecoderSC(nn.Module):
    """Simplified ResNet-style Semantic Decoder for image reconstruction."""
    def __init__(self, arch_name: str, latent_dim: int, output_channels: int = 3):
        """
        Args:
            arch_name (str): Name of the corresponding ResNet encoder architecture
                             (used to determine the feature dim before projection).
            latent_dim (int): The dimension of the input latent representation 'z_prime'.
            output_channels (int): Number of channels in the output image (e.g., 3 for RGB).
        """
        super().__init__()
        # Determine the feature dimension the encoder backbone produced *before* projection
        if arch_name in ['resnet18', 'resnet34']:
            feature_dim = 512
            initial_upsample_depth = 512 # Depth of the first upsampling block
            # Starting spatial size after initial Linear -> Reshape
            # For CIFAR (32x32), ResNet18/34 output before avgpool is (B, 512, 1, 1)
            # We want to reverse this. Start small, e.g., 4x4.
            self.initial_wh = 4
        elif arch_name == 'resnet50':
             feature_dim = 2048
             initial_upsample_depth = 1024 # Reduce depth gradually
             self.initial_wh = 4
        else:
             raise ValueError(f"Unsupported ResNet architecture for decoder: {arch_name}")

        self.latent_dim = latent_dim
        self.output_channels = output_channels

        # 1. Project latent vector back to a feature map shape
        self.fc_upsample = nn.Linear(latent_dim, initial_upsample_depth * self.initial_wh * self.initial_wh)

        # 2. Transposed Convolution layers to upsample (mirroring ResNet stages)
        # This part needs careful design and experimentation. Example for ResNet18/34 structure:
        # Output size = (Input size - 1) * Stride - 2 * Padding + Kernel Size + Output Padding
        self.upsample_layers = nn.Sequential(
            # From 4x4 -> 8x8
            self._make_upsample_block(initial_upsample_depth, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            # From 8x8 -> 16x16
            self._make_upsample_block(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # From 16x16 -> 32x32
            self._make_upsample_block(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            # Final layer to get desired output channels (32x32)
            nn.ConvTranspose2d(64, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh() # Output pixel values typically in [-1, 1] if input was normalized to [-1, 1]
                      # Or Sigmoid if [0, 1]. Match the normalization range.
                      # Tanh is common with the normalization used in get_cifar_transforms
        )
        print(f"Initialized ResNetDecoderSC (arch={arch_name}, latent_dim={latent_dim})")


    def _make_upsample_block(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, activation=True, batchnorm=True):
        """Helper to create an upsampling block."""
        layers = OrderedDict()
        layers['convtranspose'] = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=not batchnorm)
        if batchnorm:
            layers['batchnorm'] = nn.BatchNorm2d(out_channels)
        if activation:
             layers['relu'] = nn.ReLU(inplace=True)
        return nn.Sequential(layers)


    def forward(self, z_prime: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent representation 'z_prime' back into an image.

        Args:
            z_prime (torch.Tensor): Input latent tensor (batch_size, latent_dim).

        Returns:
            torch.Tensor: Reconstructed image tensor (batch_size, output_channels, H, W).
                          For CIFAR reconstruction, H=W=32.
        """
        # Project and reshape
        x = self.fc_upsample(z_prime)
        # print("Decoder fc out shape:", x.shape) # Debug
        x = x.view(x.size(0), -1, self.initial_wh, self.initial_wh) # Reshape to (B, C, initial_wh, initial_wh)
        # print("Decoder reshaped shape:", x.shape) # Debug

        # Upsample through transposed conv layers
        output_image = self.upsample_layers(x)
        # print("Decoder final output shape:", output_image.shape) # Debug
        return output_image


# Example usage (for testing this file independently)
if __name__ == '__main__':
    # --- Test Encoder ---
    print("Testing Encoder...")
    dummy_image = torch.randn(4, 3, 32, 32) # Batch of 4 CIFAR images
    latent_dimension = 64
    encoder = ResNetEncoderSC(arch_name='resnet18', latent_dim=latent_dimension, pretrained=False)
    z = encoder(dummy_image)
    print(f"Input shape: {dummy_image.shape}")
    print(f"Output latent shape (z): {z.shape}") # Should be [4, latent_dimension]
    assert z.shape == (4, latent_dimension)

    # --- Test Decoder ---
    print("\nTesting Decoder...")
    dummy_latent = torch.randn(4, latent_dimension) # Batch of 4 latent vectors
    decoder = ResNetDecoderSC(arch_name='resnet18', latent_dim=latent_dimension, output_channels=3)
    reconstructed_image = decoder(dummy_latent)
    print(f"Input latent shape (z'): {dummy_latent.shape}")
    print(f"Output reconstructed shape: {reconstructed_image.shape}") # Should be [4, 3, 32, 32]
    assert reconstructed_image.shape == (4, 3, 32, 32)

    print("\nTesting with ResNet50...")
    encoder50 = ResNetEncoderSC(arch_name='resnet50', latent_dim=128, pretrained=False)
    decoder50 = ResNetDecoderSC(arch_name='resnet50', latent_dim=128)
    z50 = encoder50(dummy_image)
    recon50 = decoder50(z50)
    print(f"ResNet50 Latent shape: {z50.shape}")
    print(f"ResNet50 Recon shape: {recon50.shape}")
    assert z50.shape == (4, 128)
    assert recon50.shape == (4, 3, 32, 32)