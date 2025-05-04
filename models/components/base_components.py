# models/components/base_components.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResidualBlock # 从同一目录下的 blocks 导入

# --- 发射器 (编码器组件) ---
class BaseTransmitter(nn.Module):
    """基于表格结构的基础发射器。"""
    def __init__(self, input_channels=3, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # 输入: (B, 3, 32, 32) 假设 CIFAR-10/100

        # 层组 1: (卷积层 + ReLU) x 2 -> 输出: 128 x 16 x 16
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        ) # 输出: (B, 128, 16, 16)

        # 层组 2: 残差块 -> 输出: 128 x 16 x 16
        self.res_block1 = ResidualBlock(128) # 输出: (B, 128, 16, 16)

        # 层组 3: (卷积层 + ReLU) x 3 -> 输出: 8 x 4 x 4
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 8x8 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 8, kernel_size=3, stride=1, padding=1),   # 4x4 -> 4x4, 128 -> 8 通道
            nn.ReLU(inplace=True)
        ) # 输出: (B, 8, 4, 4)

        # 层组 4: Reshape + Dense + Tanh -> 输出: 128 (latent_dim)
        self.final_block = nn.Sequential(
            nn.Flatten(), # 输入: (B, 8*4*4 = 128)
            nn.Linear(8 * 4 * 4, latent_dim), # 128 -> 128
            nn.Tanh() # 输出范围 [-1, 1]
        ) # 输出: (B, 128)

        print(f"Initialized BaseTransmitter (latent_dim={latent_dim})")

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.res_block1(x)
        x = self.conv_block2(x)
        z = self.final_block(x)
        return z

# --- 接收器 (分类器组件) ---
class BaseReceiverClassifier(nn.Module):
    """基于表格结构的基础分类器接收器。"""
    def __init__(self, latent_dim=128, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.reshape_target_channels = 8
        self.reshape_target_spatial = 4

        # 层 1: Dense + ReLU + Reshape -> 输出: 8 x 4 x 4
        self.fc_reshape = nn.Sequential(
            nn.Linear(latent_dim, self.reshape_target_channels * self.reshape_target_spatial * self.reshape_target_spatial), # 128 -> 128
            nn.ReLU(inplace=True)
        ) # Reshape 后输出: (B, 8, 4, 4)

        # 层 2: 卷积层 + ReLU -> 输出: 512 x 4 x 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.reshape_target_channels, 512, kernel_size=3, stride=1, padding=1), # 8 -> 512 通道
            nn.ReLU(inplace=True)
        ) # 输出: (B, 512, 4, 4)

        # 层 3: 残差块 -> 输出: 512 x 4 x 4
        self.res_block1 = ResidualBlock(512) # 输出: (B, 512, 4, 4)

        # 层 4: 池化 -> 输出: 512
        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # (B, 512, 1, 1)
            nn.Flatten()                  # (B, 512)
        ) # 输出: (B, 512)

        # 层 5: Dense (输出 Logits)
        self.fc_final = nn.Linear(512, num_classes) # 输出: (B, num_classes)

        print(f"Initialized BaseReceiverClassifier (num_classes={num_classes})")

    def forward(self, z_prime):
        x = self.fc_reshape(z_prime)
        x = x.view(x.size(0), self.reshape_target_channels, self.reshape_target_spatial, self.reshape_target_spatial) # Reshape
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.pooling(x)
        logits = self.fc_final(x)
        return logits

# --- 接收器 (解码器/重建组件) ---
class BaseReceiverDecoder(nn.Module):
    """基于表格结构的基础解码器接收器 (Tanh 输出)。"""
    def __init__(self, latent_dim=128, output_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.reshape_target_channels = 8
        self.reshape_target_spatial = 4

        # 层 1: Dense + Tanh + Reshape -> 输出: 8 x 4 x 4
        self.fc_reshape = nn.Sequential(
            nn.Linear(latent_dim, self.reshape_target_channels * self.reshape_target_spatial * self.reshape_target_spatial), # 128 -> 128
            nn.Tanh() # 这里使用 Tanh 激活
        ) # Reshape 后输出: (B, 8, 4, 4)

        # 层 2: 卷积层 + ReLU -> 输出: 512 x 4 x 4
        self.conv1 = nn.Sequential(
             nn.Conv2d(self.reshape_target_channels, 512, kernel_size=3, stride=1, padding=1), # 8 -> 512 通道
             nn.ReLU(inplace=True)
        ) # 输出: (B, 512, 4, 4)

        # 层 3: (反卷积层 + ReLU) x 2 -> 输出: 128 x 16 x 16
        self.deconv_block1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 4x4->8x8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 8x8->16x16
            nn.ReLU(inplace=True)
        ) # 输出: (B, 128, 16, 16)

        # 层 4: 残差块 -> 输出: 128 x 16 x 16
        self.res_block1 = ResidualBlock(128) # 输出: (B, 128, 16, 16)

        # 层 5: 反卷积层 + ReLU -> 输出: 64 x 32 x 32
        self.deconv2 = nn.Sequential(
             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 16x16->32x32
             nn.ReLU(inplace=True)
        ) # 输出: (B, 64, 32, 32)

        # 层 6: 卷积层 + Tanh -> 输出: 3 x 32 x 32
        self.conv_final = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1), # 64 -> 3 通道
            nn.Tanh() # 使用 Tanh 使输出范围为 [-1, 1]
        ) # 输出: (B, 3, 32, 32)

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