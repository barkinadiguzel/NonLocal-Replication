import torch.nn as nn
from ..backbone.resnet_blocks import BasicResBlock
from ..nonlocal.nonlocal_block import NonLocalBlock

class NonLocalCNN(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, num_blocks=5, f_type="embedded", dim=3):
        super().__init__()
        self.dim = dim

        self.stem = BasicResBlock(in_ch, base_ch, dim=dim)

        self.nonlocals = nn.ModuleList([
            NonLocalBlock(base_ch, f_type=f_type, dim=dim) for _ in range(num_blocks)
        ])

        if dim == 3:
            self.head = nn.AdaptiveAvgPool3d(1)
        else:
            self.head = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)
        for block in self.nonlocals:
            x = block(x)
        x = self.head(x)
        return x.view(x.size(0), -1)
