import torch.nn as nn
from ..backbone.resnet_blocks import BasicResBlock, ResNetStage
from ..nonlocal.nonlocal_block import NonLocalBlock

class NonLocalCNN(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, nonlocal_positions=None, f_type="embedded", dim=3):
        super().__init__()
        nonlocal_positions = nonlocal_positions or ["res2", "res3", "res4"]

        self.stem = BasicResBlock(in_ch, base_ch, dim=dim)
        self.res2 = ResNetStage(base_ch, base_ch, num_blocks=2, dim=dim)
        self.res3 = ResNetStage(base_ch, base_ch*2, num_blocks=2, dim=dim)
        self.res4 = ResNetStage(base_ch*2, base_ch*4, num_blocks=2, dim=dim)
        
        self.nl_res2 = NonLocalBlock(base_ch, f_type=f_type, dim=dim) if "res2" in nonlocal_positions else None
        self.nl_res3 = NonLocalBlock(base_ch*2, f_type=f_type, dim=dim) if "res3" in nonlocal_positions else None
        self.nl_res4 = NonLocalBlock(base_ch*4, f_type=f_type, dim=dim) if "res4" in nonlocal_positions else None

        self.head = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.res2(x)
        if self.nl_res2: x = self.nl_res2(x)
        x = self.res3(x)
        if self.nl_res3: x = self.nl_res3(x)
        x = self.res4(x)
        if self.nl_res4: x = self.nl_res4(x)
        x = self.head(x)
        return x.view(x.size(0), -1)
