# models/sc_model.py

import torch
import torch.nn as nn
from typing import Union, Tuple, Dict, Any

# 从 SC 包导入 channel (路径相对于项目根目录)
from SC.channel import get_channel
# 从同级目录下的 components 导入
from .components.base_components import BaseTransmitter, BaseReceiverClassifier, BaseReceiverDecoder
from .components.resnet_components import ResNetEncoderSC, ResNetDecoderSC, MLPClassifierHead

class SC_Model(nn.Module):
    """
    统一的语义通信系统模型。
    根据配置选择 Base 或 ResNet 组件。
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化端到端语义通信模型。

        Args:
            config (dict): 包含所有必要配置的字典。
                           需要 'victim_model', 'channel', 'task', 'dataset' 键。
                           'victim_model' 子字典必须包含 'type' ('base_model' 或 'resnet_sc')
                           以及 'encoder' 和 'decoder' 的相应配置。
        """
        super().__init__()

        # 解析配置
        try:
            victim_config = config['victim_model']
            channel_config = config['channel']
            self.task = config['task'].lower()
            dataset_name = config['dataset']['name']
            model_type = victim_config['type'].lower() # 强制要求 type 字段
        except KeyError as e:
            raise KeyError(f"Configuration error: Missing required key - {e}. "
                           "Ensure 'victim_model', 'channel', 'task', 'dataset', "
                           "and 'victim_model.type' are present.")

        if self.task not in ['reconstruction', 'classification']:
            raise ValueError(f"Unsupported task: {self.task}. Choose 'reconstruction' or 'classification'.")
        if model_type not in ['base_model', 'resnet_sc']:
             raise ValueError(f"Unsupported victim_model.type: {model_type}. Choose 'base_model' or 'resnet_sc'.")

        # 获取通用参数
        encoder_config = victim_config['encoder']
        decoder_config = victim_config['decoder']
        latent_dim = encoder_config['latent_dim']
        num_classes = 10 if dataset_name == 'cifar10' else 100 # 默认类别数

        # --- 1. 初始化编码器 ---
        if model_type == 'base_model':
            self.encoder = BaseTransmitter(
                input_channels=encoder_config.get('input_channels', 3),
                latent_dim=latent_dim
            )
        elif model_type == 'resnet_sc':
            # 确保 arch_name 存在
            if 'arch_name' not in encoder_config:
                 raise KeyError("Configuration error: 'victim_model.encoder.arch_name' is required for type 'resnet_sc'.")
            self.encoder = ResNetEncoderSC(
                arch_name=encoder_config['arch_name'],
                latent_dim=latent_dim,
                pretrained=encoder_config.get('pretrained', False) # 默认不预训练
            )

        # --- 2. 初始化信道 ---
        self.channel = get_channel(channel_config)

        # --- 3. 初始化解码器 ---
        if self.task == 'reconstruction':
            output_channels = decoder_config.get('output_channels', 3)
            if model_type == 'base_model':
                self.decoder = BaseReceiverDecoder(
                    latent_dim=latent_dim,
                    output_channels=output_channels
                )
            elif model_type == 'resnet_sc':
                 # 确保 arch_name 存在 (ResNetDecoderSC 可能需要它来确定结构)
                if 'arch_name' not in decoder_config:
                     raise KeyError("Configuration error: 'victim_model.decoder.arch_name' is required for type 'resnet_sc' reconstruction decoder.")
                self.decoder = ResNetDecoderSC(
                    arch_name=decoder_config['arch_name'],
                    latent_dim=latent_dim,
                    output_channels=output_channels
                )
        elif self.task == 'classification':
            # 获取类别数，如果配置中没有，则使用默认值
            num_classes_actual = decoder_config.get('num_classes', num_classes)
            if model_type == 'base_model':
                 self.decoder = BaseReceiverClassifier(
                     latent_dim=latent_dim,
                     num_classes=num_classes_actual
                 )
            elif model_type == 'resnet_sc':
                 self.decoder = MLPClassifierHead(
                     latent_dim=latent_dim,
                     num_classes=num_classes_actual,
                     dropout=decoder_config.get('dropout', 0.5) # 从配置获取 dropout
                 )

        print(f"Initialized unified SC_Model (Type: {model_type}, Task: {self.task})")


    def forward(self, x: torch.Tensor, return_latent: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        执行端到端的语义通信过程。

        Args:
            x (torch.Tensor): 输入数据张量 (例如, 图像批次)。
            return_latent (bool): 如果为 True，则同时返回潜变量 z 和 z_prime。

        Returns:
            torch.Tensor | tuple:
                如果 return_latent 为 False: 最终输出 Y (重建图像或分类 logits)。
                如果 return_latent 为 True: 元组 (Y, z, z_prime)。
        """
        # 1. 编码
        z = self.encoder(x)

        # 2. 通过信道传输
        # 确保信道与模型整体处于相同的训练/评估模式
        self.channel.train(self.training)
        z_prime = self.channel(z)

        # 3. 解码
        y = self.decoder(z_prime)

        if return_latent:
            return y, z, z_prime
        else:
            return y

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """直接调用编码器。"""
        return self.encoder(x)

    def decode(self, z_prime: torch.Tensor) -> torch.Tensor:
         """直接调用解码器（确保模式正确）。"""
         # 如果解码器有 Dropout 等层，确保它获得正确的模式
         self.decoder.train(self.training)
         return self.decoder(z_prime)