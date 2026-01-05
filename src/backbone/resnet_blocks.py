import torch.nn as nn
from ..layers.conv_layer import ConvBNActivation

class BasicResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dim=3, stride=1):
        super().__init__()
        self.conv1 = ConvBNActivation(in_ch, out_ch, kernel_size=3, dim=dim, stride=stride, padding=1)
        self.conv2 = ConvBNActivation(out_ch, out_ch, kernel_size=3, dim=dim, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x
