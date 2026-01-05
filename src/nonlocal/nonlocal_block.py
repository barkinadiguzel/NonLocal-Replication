import torch
import torch.nn as nn
from .f_functions import gaussian, embedded_gaussian, dot_product, concat_function
from .g_functions import LinearEmbedding

class NonLocalBlock(nn.Module):
    def __init__(self, in_ch, inter_ch=None, f_type="embedded", dim=3):
        super().__init__()
        self.dim = dim
        inter_ch = inter_ch or in_ch // 2

        self.g = LinearEmbedding(in_ch, inter_ch)
        self.Wz = nn.Conv3d(inter_ch, in_ch, kernel_size=1)
        nn.init.constant_(self.Wz.weight, 0)

        if f_type == "gaussian":
            self.f = gaussian
        elif f_type == "embedded":
            self.f = embedded_gaussian
        elif f_type == "dot":
            self.f = dot_product
        else:
            raise ValueError(f"{f_type} f-function not implemented")

    def forward(self, x):
        batch, C, T, H, W = x.shape
        g_x = self.g(x).view(batch, -1, T*H*W)   
        x_flat = x.view(batch, C, -1)
        f_x = self.f(x_flat.transpose(1,2), x_flat.transpose(1,2))
        f_x = f_x / f_x.sum(-1, keepdim=True)
        y = torch.matmul(f_x, g_x.transpose(1,2)).transpose(1,2).view(batch, C, T, H, W)
        return self.Wz(y) + x
