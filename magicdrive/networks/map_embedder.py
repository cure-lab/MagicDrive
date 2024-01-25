from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.controlnet import zero_module


class BEVControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int = 320,
        conditioning_size: Tuple[int, int, int] = (25, 200, 200),  # only use 25
        block_out_channels: Tuple[int] = (32, 64, 128, 256),
    ):
        super().__init__()
        # input size   25, 200, 200
        # output size 320,  28,  50

        self.conv_in = nn.Conv2d(
            conditioning_size[0],
            block_out_channels[0],
            kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 2):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                nn.Conv2d(
                    channel_in, channel_out, kernel_size=3, padding=(2, 1),
                    stride=2))
        channel_in = block_out_channels[-2]
        channel_out = block_out_channels[-1]
        self.blocks.append(
            nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=(2, 1))
        )
        self.blocks.append(
            nn.Conv2d(
                channel_in, channel_out, kernel_size=3, padding=(2, 1),
                stride=(2, 1)))

        self.conv_out = zero_module(
            nn.Conv2d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding
