# models/components/resnet_components.py
import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
# 注意：如果 ResNetDecoderSC 未来需要 ResidualBlock，也需要从 .blocks 导入

# 辅助函数：获取预训练的 ResNet 模型骨干（移除最后的 FC 层）
def _get_resnet_backbone(arch_name: str, pretrained: bool = True):
    """加载预训练的 ResNet 模型并移除最后的分类层。"""
    weights = None
    if pretrained:
        if arch_name == 'resnet18': weights = models.ResNet18_Weights.DEFAULT
        elif arch_name == 'resnet34': weights = models.ResNet34_Weights.DEFAULT
        elif arch_name == 'resnet50': weights = models.ResNet50_Weights.DEFAULT
        # ... 添加更多架构的权重
        else: print(f"Warning: No default pretrained weights specified for {arch_name}. Using random init.")
    
    if arch_name == 'resnet18':
        model = models.resnet18(weights=weights)
        feature_dim = 512 # ResNet18/34 输出特征维度
    elif arch_name == 'resnet34':
        model = models.resnet34(weights=weights)
        feature_dim = 512
    elif arch_name == 'resnet50':
         model = models.resnet50(weights=weights)
         feature_dim = 2048 # ResNet50+ 输出特征维度
    # 添加更多架构支持
    else:
        raise ValueError(f"Unsupported ResNet architecture: {arch_name}")

    # 移除最后的 FC 层
    modules = list(model.children())[:-1] # 移除 model.fc
    backbone = nn.Sequential(*modules)
    return backbone, feature_dim

# --- ResNet 编码器 ---
class ResNetEncoderSC(nn.Module):
    """基于 ResNet 的语义编码器。"""
    def __init__(self, arch_name: str, latent_dim: int, pretrained: bool = True):
        super().__init__()
        self.backbone, feature_dim = _get_resnet_backbone(arch_name, pretrained)
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # 确保空间维度是 1x1
            nn.Flatten(),                 # 展平成向量
            nn.Linear(feature_dim, latent_dim),
            # nn.Tanh() # 根据需要可选 Tanh
        )
        print(f"Initialized ResNetEncoderSC (arch={arch_name}, latent_dim={latent_dim}, pretrained={pretrained})")
        print(f"  Backbone output features: {feature_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        z = self.projection(features)
        return z

# --- ResNet 解码器 (重建) ---
class ResNetDecoderSC(nn.Module):
    """简化的 ResNet 风格语义解码器，用于图像重建。"""
    def __init__(self, arch_name: str, latent_dim: int, output_channels: int = 3):
        super().__init__()
        # 确定编码器骨干在投影前的特征维度
        if arch_name in ['resnet18', 'resnet34']:
            # 对于 CIFAR (32x32), ResNet18/34 在 avgpool 前的输出是 (B, 512, 1, 1)
            # ResNet50 是 (B, 2048, 1, 1)
            # 我们需要反转这个过程。这里简化了，直接从 latent_dim 开始上采样。
            # 更复杂的设计会先映射回 backbone feature_dim * H * W
            initial_upsample_depth = 512 # 第一个上采样块的深度
            self.initial_wh = 4 # 假设从 4x4 开始上采样
        elif arch_name == 'resnet50':
             initial_upsample_depth = 1024 # 逐渐减少深度
             self.initial_wh = 4
        else:
             raise ValueError(f"Unsupported ResNet architecture for decoder: {arch_name}")

        self.latent_dim = latent_dim
        self.output_channels = output_channels

        # 1. 将潜在向量映射回特征图形状
        self.fc_upsample = nn.Linear(latent_dim, initial_upsample_depth * self.initial_wh * self.initial_wh)

        # 2. 使用转置卷积层进行上采样
        self.upsample_layers = nn.Sequential(
            # 从 4x4 -> 8x8
            self._make_upsample_block(initial_upsample_depth, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            # 从 8x8 -> 16x16
            self._make_upsample_block(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # 从 16x16 -> 32x32
            self._make_upsample_block(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            # 最终层得到所需输出通道 (32x32)
            nn.ConvTranspose2d(64, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh() # 输出像素值范围 [-1, 1]
        )
        print(f"Initialized ResNetDecoderSC (arch={arch_name}, latent_dim={latent_dim})")

    def _make_upsample_block(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, activation=True, batchnorm=True):
        """创建上采样块的辅助函数。"""
        layers = OrderedDict()
        layers['convtranspose'] = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=not batchnorm)
        if batchnorm:
            layers['batchnorm'] = nn.BatchNorm2d(out_channels)
        if activation:
             layers['relu'] = nn.ReLU(inplace=True)
        return nn.Sequential(layers)

    def forward(self, z_prime: torch.Tensor) -> torch.Tensor:
        x = self.fc_upsample(z_prime)
        x = x.view(x.size(0), -1, self.initial_wh, self.initial_wh)
        output_image = self.upsample_layers(x)
        return output_image

# --- MLP 分类头 ---
class MLPClassifierHead(nn.Module):
    """用于分类任务的简单 MLP 解码器/头。"""
    def __init__(self, latent_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        print(f"Initialized MLPClassifierHead (latent_dim={latent_dim}, num_classes={num_classes})")

    def forward(self, z_prime: torch.Tensor) -> torch.Tensor:
        return self.head(z_prime)