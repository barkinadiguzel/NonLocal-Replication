import torch.nn as nn

def get_norm(norm_type="batch", dim=2, num_features=None):
    if norm_type.lower() == "batch":
        return nn.BatchNorm2d(num_features) if dim == 2 else nn.BatchNorm3d(num_features)
    elif norm_type.lower() == "layer":
        return nn.LayerNorm(num_features)
    else:
        raise ValueError(f"{norm_type} normalization not implemented")
