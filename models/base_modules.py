import torch
import torch.nn as nn
from typing import Tuple


class BasicConv3D(nn.Module):
    """Basic 3D convolution block"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int, int],
                 stride: Tuple[int, int, int] = (1, 1, 1),
                 padding: Tuple[int, int, int] = (0, 0, 0),
                 groups: int = 1,
                 bias: bool = False):
        super(BasicConv3D, self).__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelShuffle(nn.Module):
    """Channel shuffle operation"""

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, depth, height, width = x.size()
        channels_per_group = num_channels // self.groups

        # Reshape
        x = x.view(batchsize, self.groups, channels_per_group, depth, height, width)

        # Permute
        x = x.permute(0, 2, 1, 3, 4, 5).contiguous()

        # Flatten
        x = x.view(batchsize, num_channels, depth, height, width)

        return x


class FastToSlowFusion(nn.Module):
    """Basic fusion module from Fast to Slow pathway"""

    def __init__(self,
                 in_channels: int,
                 fusion_kernel: int = 5,
                 alpha: int = 8):
        super(FastToSlowFusion, self).__init__()
        self.conv = nn.Conv3d(
            in_channels,
            in_channels * 2,
            kernel_size=(fusion_kernel, 1, 1),
            stride=(alpha, 1, 1),
            padding=(fusion_kernel // 2, 0, 0),
            bias=False
        )
        self.bn = nn.BatchNorm3d(in_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# Basic building block for pathway
class PathwayBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: Tuple[int, int, int] = (1, 1, 1)):
        super(PathwayBlock, self).__init__()
        self.conv1 = BasicConv3D(in_channels, out_channels,
                                 kernel_size=(1, 1, 1),
                                 stride=(1, 1, 1))
        self.conv2 = BasicConv3D(out_channels, out_channels,
                                 kernel_size=(3, 3, 3),
                                 stride=stride,
                                 padding=(1, 1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x