# models/components/blocks.py
import torch
import torch.nn as nn

# 基础残差块 (保持维度不变)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 使用 kernel_size=3, padding=1, stride=1 来维持空间维度
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True) # 在跳跃连接相加后应用 ReLU

    def forward(self, x):
        residual = x
        out = self.relu1(self.conv1(x))
        out = self.conv2(out)
        out += residual # 跳跃连接
        out = self.relu2(out) # 相加后的最终 ReLU
        return out

