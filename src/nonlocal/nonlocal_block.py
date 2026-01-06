import torch
import torch.nn as nn
import torch.nn.functional as F

class SpacetimeNonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super().__init__()
        # Bottleneck channels
        inter_channels = inter_channels or in_channels // 2

        # g(x) embedding
        self.g = nn.Conv3d(in_channels, inter_channels, kernel_size=1, bias=False)

        # θ ve φ embedding
        self.theta = nn.Conv3d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.phi = nn.Conv3d(in_channels, inter_channels, kernel_size=1, bias=False)

        # Output transform
        self.Wz = nn.Conv3d(inter_channels, in_channels, kernel_size=1, bias=False)
        nn.init.constant_(self.Wz.weight, 0) 

    def forward(self, x):
        """
        x: [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape

        # Embedding
        g_x = self.g(x).view(B, -1, T*H*W)        
        g_x = g_x.permute(0, 2, 1)               

        theta_x = self.theta(x).view(B, -1, T*H*W)  
        theta_x = theta_x.permute(0, 2, 1)         

        phi_x = self.phi(x).view(B, -1, T*H*W)      

        # f(θ, φ) = softmax(θ^T φ)
        f = torch.matmul(theta_x, phi_x)            
        f_div_C = F.softmax(f, dim=-1)             

        # y = f * g(x)
        y = torch.matmul(f_div_C, g_x)             
        y = y.permute(0, 2, 1).contiguous().view(B, -1, T, H, W) 

        # Output + residual
        z = self.Wz(y) + x
        return z
