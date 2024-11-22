import torch
import torch.nn as nn
from typing import Optional, List, Tuple


def channel_shuffle(x, groups):
    '''Channel shuffle operation
    Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
    batchsize, num_channels, depth, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups,
               channels_per_group, depth, height, width)
    x = x.permute(0, 2, 1, 3, 4, 5).contiguous()
    x = x.view(batchsize, num_channels, depth, height, width)
    return x


class MobileOneBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: Tuple = (1, 1, 1),
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1) -> None:
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        self.se = nn.Identity()
        self.activation = nn.ReLU()

        if inference_mode:
            self.reparam_conv = nn.Conv3d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=True)
        else:
            self.rbr_skip = nn.BatchNorm3d(num_features=in_channels) \
                if out_channels == in_channels and stride == (1, 1, 1) else None

            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size,
                                              padding=padding))
            self.rbr_conv = nn.ModuleList(rbr_conv)

            self.rbr_scale = None
            if kernel_size > 1:
                self.rbr_scale = self._conv_bn(kernel_size=1,
                                               padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def reparameterize(self):
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv3d(in_channels=self.rbr_conv[0].conv.in_channels,
                                      out_channels=self.rbr_conv[0].conv.out_channels,
                                      kernel_size=self.rbr_conv[0].conv.kernel_size,
                                      stride=self.rbr_conv[0].conv.stride,
                                      padding=self.rbr_conv[0].conv.padding,
                                      dilation=self.rbr_conv[0].conv.dilation,
                                      groups=self.rbr_conv[0].conv.groups,
                                      bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad, pad, pad])

        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm3d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                    self.kernel_size // 2,
                                    self.kernel_size // 2,
                                    self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self,
                 kernel_size: int,
                 padding: int) -> nn.Sequential:
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv3d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,
                                              bias=False))
        mod_list.add_module('bn', nn.BatchNorm3d(num_features=self.out_channels))
        return mod_list


class MobileOneBlocklist(nn.Module):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 num_blocks: int = 1,
                 stride: Tuple = (1, 1, 1),
                 use_se: bool = False,
                 inference_mode: bool = False,
                 num_conv_branches: int = 1,
                 num_se_blocks: int = 0):
        super(MobileOneBlocklist, self).__init__()

        self.GConv1 = MobileOneBlock(in_channels=in_planes,
                                     out_channels=in_planes,
                                     kernel_size=1,
                                     stride=(1, 1, 1),
                                     padding=0,
                                     groups=1,
                                     inference_mode=inference_mode,
                                     use_se=use_se,
                                     num_conv_branches=num_conv_branches)
        self.DWConv = MobileOneBlock(in_channels=in_planes,
                                     out_channels=in_planes,
                                     kernel_size=3,
                                     stride=stride,
                                     padding=1,
                                     groups=in_planes,
                                     inference_mode=inference_mode,
                                     use_se=use_se,
                                     num_conv_branches=num_conv_branches)
        self.GConv2 = MobileOneBlock(in_channels=in_planes,
                                     out_channels=out_planes,
                                     kernel_size=1,
                                     stride=(1, 1, 1),
                                     padding=0,
                                     groups=1,
                                     inference_mode=inference_mode,
                                     use_se=use_se,
                                     num_conv_branches=num_conv_branches)

    def forward(self, input):
        out = self.GConv1(input)
        out = channel_shuffle(out, 2)
        out = self.DWConv(out)
        out = self.GConv2(out)
        return out