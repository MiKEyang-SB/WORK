import numpy as np
import einops
import torch
import torch.nn.functional as F

def normalise_quat(x: torch.Tensor):
    return x / torch.clamp(x.square().sum(dim=-1).sqrt().unsqueeze(-1), min=1e-10)
#对四元数进行归一化
