import torch.nn as nn

class LinearEmbedding(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.Wg = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        return self.Wg(x)
