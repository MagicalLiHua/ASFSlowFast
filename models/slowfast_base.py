import torch
import torch.nn as nn
from models.base_modules import MobileOneBlocklist


class BasicConv3D(nn.Module):
    """Basic 3D convolution block for replacing Depthwise_Separable"""

    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False):
        super(BasicConv3D, self).__init__()
        self.conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.conv(input)
        out = self.bn(out)
        out = self.relu(out)
        return out


class BasicLateral(nn.Module):
    """Basic lateral connection block"""

    def __init__(self, channels):
        super(BasicLateral, self).__init__()
        self.conv = nn.Conv3d(
            in_channels=channels,
            out_channels=channels * 2,
            kernel_size=(5, 1, 1),
            stride=(4, 1, 1),
            padding=(2, 0, 0),
            bias=False
        )
        self.bn = nn.BatchNorm3d(channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


# Default parameters
default_params = {
    'width_multipliers': (0.5, 0.5, 0.5, 0.5),
    'num_feature': 6,
    'lateral': False,
    'num_classes': 12,
    'actions_classes': 6,
    'users_classes': 15,
    'epoch_num': 150,
    'batch_size': 16,
    'learning_rate': 0.01,
}


class SlowFast(nn.Module):
    def __init__(self, actclass_num=default_params['actions_classes'],
                 userclass_num=default_params['users_classes'],
                 dropout=0.5,
                 width_multipliers=default_params['width_multipliers'],
                 C_multipliers=(0.5,)):
        super(SlowFast, self).__init__()

        # Fast path definition
        self.fast_conv1 = BasicConv3D(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # Fast path main blocks
        self.fast_shu2_1 = MobileOneBlocklist(in_planes=8, out_planes=int(32 * width_multipliers[0]), stride=(1, 2, 2))
        self.fast_shu3_1 = MobileOneBlocklist(int(32 * width_multipliers[0]), out_planes=int(64 * width_multipliers[1]),
                                              stride=(1, 2, 2))
        self.fast_shu4_1 = MobileOneBlocklist(in_planes=int(64 * width_multipliers[1]),
                                              out_planes=int(128 * width_multipliers[2]), stride=(1, 2, 2))
        self.fast_shu5 = MobileOneBlocklist(in_planes=int(128 * width_multipliers[2]),
                                            out_planes=int(256 * width_multipliers[3]))

        # Lateral connections with basic implementation
        self.lateral_p1 = BasicLateral(8)
        self.lateral_res2 = BasicLateral(int(32 * width_multipliers[0]))
        self.lateral_res3 = BasicLateral(int(64 * width_multipliers[1]))
        self.lateral_res4 = BasicLateral(int(128 * width_multipliers[2]))

        # Slow path definition
        self.slow_conv1 = BasicConv3D(3, int(64 * C_multipliers[0]), kernel_size=(1, 7, 7), stride=(1, 2, 2),
                                      padding=(0, 3, 3), bias=False)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # Slow path main blocks
        self.slow_shu2_1 = MobileOneBlocklist(in_planes=int(16 + (64 * C_multipliers[0])),
                                              out_planes=int(256 * width_multipliers[0] * C_multipliers[0]),
                                              stride=(1, 2, 2))
        self.slow_shu3_1 = MobileOneBlocklist(
            in_planes=int((64 * width_multipliers[0]) + 256 * width_multipliers[0] * C_multipliers[0]),
            out_planes=int(512 * width_multipliers[1] * C_multipliers[0]),
            stride=(1, 2, 2))
        self.slow_shu4_1 = MobileOneBlocklist(
            in_planes=int((128 * width_multipliers[1]) + 512 * width_multipliers[0] * C_multipliers[0]),
            out_planes=int(1024 * width_multipliers[2] * C_multipliers[0]),
            stride=(1, 2, 2))
        self.slow_shu5 = MobileOneBlocklist(
            in_planes=int((256 * width_multipliers[2]) + 1024 * width_multipliers[2] * C_multipliers[0]),
            out_planes=int(2048 * width_multipliers[3] * C_multipliers[0]))

        # Output layers
        self.dp = nn.Dropout(dropout)
        self.fcx = nn.Linear(int(256 * width_multipliers[3] + (2048 * width_multipliers[3] * C_multipliers[0])),
                             actclass_num, bias=True)
        self.fcu = nn.Linear(int(256 * width_multipliers[3] + (2048 * width_multipliers[3] * C_multipliers[0])),
                             userclass_num, bias=True)

    def forward(self, input):
        # Process fast pathway
        fast, lateral = self.FastPath(input[:, :, ::4, :, :])  # temporal stride 4
        # Process slow pathway
        slow = self.SlowPath(input[:, :, ::16, :, :], lateral)  # temporal stride 16
        # Combine pathways
        x = torch.cat([slow, fast], dim=1)

        # Output heads
        a = self.dp(x)
        a = self.fcx(a)

        return a

    def SlowPath(self, input, lateral):
        x = self.slow_conv1(input)
        x = self.slow_maxpool(x)
        x = torch.cat([x, lateral[0]], dim=1)

        x = self.slow_shu2_1(x)
        x = torch.cat([x, lateral[1]], dim=1)

        x = self.slow_shu3_1(x)
        x = torch.cat([x, lateral[2]], dim=1)

        x = self.slow_shu4_1(x)
        x = torch.cat([x, lateral[3]], dim=1)

        x = self.slow_shu5(x)
        x = nn.AdaptiveAvgPool3d(1)(x)
        x = x.view(-1, x.size(1))
        return x

    def FastPath(self, input):
        lateral = []

        x = self.fast_conv1(input)
        pool1 = self.fast_maxpool(x)

        lateral_p = self.lateral_p1(pool1)
        lateral.append(lateral_p)  # First fusion

        shuf2_1 = self.fast_shu2_1(pool1)
        lateral_res2 = self.lateral_res2(shuf2_1)
        lateral.append(lateral_res2)  # Second fusion

        shuf3_1 = self.fast_shu3_1(shuf2_1)
        lateral_res3 = self.lateral_res3(shuf3_1)
        lateral.append(lateral_res3)  # Third fusion

        shuf4_1 = self.fast_shu4_1(shuf3_1)
        lateral_res4 = self.lateral_res4(shuf4_1)
        lateral.append(lateral_res4)  # Fourth fusion

        shuf5 = self.fast_shu5(shuf4_1)
        x = nn.AdaptiveAvgPool3d(1)(shuf5)
        x = x.view(-1, x.size(1))

        return x, lateral