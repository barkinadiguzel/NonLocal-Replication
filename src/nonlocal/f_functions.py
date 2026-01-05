import torch
import torch.nn.functional as F

def gaussian(xi, xj):
    return torch.exp(torch.matmul(xi, xj.transpose(-2, -1)))

def embedded_gaussian(xi, xj):
    return torch.exp(torch.matmul(xi, xj.transpose(-2, -1))) 

def dot_product(xi, xj):
    return torch.matmul(xi, xj.transpose(-2, -1))

def concat_function(xi, xj, wf):
    concat = torch.cat([xi.unsqueeze(2).expand(-1,-1,xj.size(1),-1),
                        xj.unsqueeze(1).expand(-1,xi.size(1),-1,-1)], dim=-1)
    return F.relu(torch.matmul(concat, wf))
