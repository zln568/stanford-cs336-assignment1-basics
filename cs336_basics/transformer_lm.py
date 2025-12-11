import torch
import numpy as np

from einops import einsum

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        std = np.sqrt(2/(in_features+out_features))
        self.weights = torch.nn.Parameter(torch.nn.init.trunc_normal_(
            torch.empty(out_features, in_features, device=device, dtype=dtype), std=std, a=-3*std, b=3*std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weights, x, "d_out d_in, ... d_in -> ... d_out")