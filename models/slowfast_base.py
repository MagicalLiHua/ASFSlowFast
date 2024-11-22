import torch
import torch.nn as nn
from typing import Tuple, List
from base_modules import BasicConv3D, ChannelShuffle, FastToSlowFusion, PathwayBlock


class SlowFastBase(nn.Module):
    """
    Basic implementation of SlowFast networks.

    Args:
        num_classes (int): Number of classes for classification
        dropout (float): Dropout rate
        width_mult (float): Width multiplier for network channels
    """

    def __init__(self,
                 num_classes: int = 400,
                 dropout: float = 0.5,
                 width_mult: float = 1.0):
        super(SlowFastBase, self).__init__()

        # Fast pathway
        self.fast_conv1 = BasicConv3D(3, int(8 * width_mult),
                                      kernel_size=(5, 7, 7),
                                      stride=(1, 2, 2),
                                      padding=(2, 3, 3))
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                         stride=(1, 2, 2),
                                         padding=(0, 1, 1))

        # Fast pathway stages
        self.fast_res2 = self._make_stage_fast(int(8 * width_mult),
                                               int(32 * width_mult),
                                               stride=(1, 2, 2))
        self.fast_res3 = self._make_stage_fast(int(32 * width_mult),
                                               int(64 * width_mult),
                                               stride=(1, 2, 2))
        self.fast_res4 = self._make_stage_fast(int(64 * width_mult),
                                               int(128 * width_mult),
                                               stride=(1, 2, 2))
        self.fast_res5 = self._make_stage_fast(int(128 * width_mult),
                                               int(256 * width_mult),
                                               stride=(1, 1, 1))

        # Lateral connections
        self.lateral1 = FastToSlowFusion(int(8 * width_mult))
        self.lateral2 = FastToSlowFusion(int(32 * width_mult))
        self.lateral3 = FastToSlowFusion(int(64 * width_mult))
        self.lateral4 = FastToSlowFusion(int(128 * width_mult))

        # Slow pathway
        self.slow_conv1 = BasicConv3D(3, int(64 * width_mult),
                                      kernel_size=(1, 7, 7),
                                      stride=(1, 2, 2),
                                      padding=(0, 3, 3))
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                         stride=(1, 2, 2),
                                         padding=(0, 1, 1))

        # Slow pathway stages
        self.slow_res2 = self._make_stage_slow(int(64 * width_mult),
                                               int(256 * width_mult),
                                               stride=(1, 2, 2))
        self.slow_res3 = self._make_stage_slow(int(256 * width_mult),
                                               int(512 * width_mult),
                                               stride=(1, 2, 2))
        self.slow_res4 = self._make_stage_slow(int(512 * width_mult),
                                               int(1024 * width_mult),
                                               stride=(1, 2, 2))
        self.slow_res5 = self._make_stage_slow(int(1024 * width_mult),
                                               int(2048 * width_mult),
                                               stride=(1, 1, 1))

        # Final layers
        self.dropout = nn.Dropout(dropout)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(int(256 * width_mult + 2048 * width_mult),
                            num_classes)

    def _make_stage_fast(self,
                         in_channels: int,
                         out_channels: int,
                         stride: Tuple[int, int, int]) -> nn.Sequential:
        """Create a stage of fast pathway"""
        return nn.Sequential(
            PathwayBlock(in_channels, out_channels, stride)
        )

    def _make_stage_slow(self,
                         in_channels: int,
                         out_channels: int,
                         stride: Tuple[int, int, int]) -> nn.Sequential:
        """Create a stage of slow pathway"""
        return nn.Sequential(
            PathwayBlock(in_channels, out_channels, stride)
        )

    def forward_fast(self, x) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass for fast pathway"""
        lateral = []

        x = self.fast_conv1(x)
        x = self.fast_maxpool(x)
        lateral.append(self.lateral1(x))

        x = self.fast_res2(x)
        lateral.append(self.lateral2(x))

        x = self.fast_res3(x)
        lateral.append(self.lateral3(x))

        x = self.fast_res4(x)
        lateral.append(self.lateral4(x))

        x = self.fast_res5(x)
        x = self.avgpool(x)
        x = x.view(-1, x.size(1))

        return x, lateral

    def forward_slow(self, x, lateral) -> torch.Tensor:
        """Forward pass for slow pathway"""
        x = self.slow_conv1(x)
        x = self.slow_maxpool(x)
        x = torch.cat([x, lateral[0]], dim=1)

        x = self.slow_res2(x)
        x = torch.cat([x, lateral[1]], dim=1)

        x = self.slow_res3(x)
        x = torch.cat([x, lateral[2]], dim=1)

        x = self.slow_res4(x)
        x = torch.cat([x, lateral[3]], dim=1)

        x = self.slow_res5(x)
        x = self.avgpool(x)
        x = x.view(-1, x.size(1))

        return x

    def forward(self, x):
        # Fast pathway - temporal stride 4
        fast_input = x[:, :, ::4, :, :]
        fast_output, lateral = self.forward_fast(fast_input)

        # Slow pathway - temporal stride 16
        slow_input = x[:, :, ::16, :, :]
        slow_output = self.forward_slow(slow_input, lateral)

        # Combine pathways
        x = torch.cat([slow_output, fast_output], dim=1)

        # Final layers
        x = self.dropout(x)
        x = self.fc(x)

        return x


def create_slowfast_base(num_classes=400, dropout=0.5, width_mult=1.0):
    """
    Create a basic SlowFast network.

    Args:
        num_classes (int): Number of classes for classification
        dropout (float): Dropout rate
        width_mult (float): Width multiplier for network channels

    Returns:
        SlowFastBase: A basic SlowFast network
    """
    model = SlowFastBase(
        num_classes=num_classes,
        dropout=dropout,
        width_mult=width_mult
    )
    return model