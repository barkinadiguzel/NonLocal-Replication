import torch
import torch.nn as nn


def conv1x1(in_channels: int, out_channels: int, dim: int = 2, bias: bool = False):
    if dim == 2:
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    elif dim == 3:
        return nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)
    else:
        raise ValueError("dim must be 2 or 3")


def conv3x3(in_channels: int, out_channels: int, dim: int = 2, stride: int = 1, padding: int = 1, bias: bool = False):
    if dim == 2:
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)
    elif dim == 3:
        return nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1), bias=bias)
    else:
        raise ValueError("dim must be 2 or 3")


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, dim=2, stride=1, padding=0):
        super().__init__()
        if dim == 2:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm2d(out_ch)
        else:
            self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
