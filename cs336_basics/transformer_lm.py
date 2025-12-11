import torch
import numpy as np

from einops import einsum, reduce

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        std = np.sqrt(2/(in_features+out_features))
        self.weights = torch.nn.Parameter(torch.nn.init.trunc_normal_(
            torch.empty(out_features, in_features, device=device, dtype=dtype), std=std, a=-3*std, b=3*std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weights, x, "d_out d_in, ... d_in -> ... d_out")
    
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.weights = torch.nn.Parameter(torch.nn.init.trunc_normal_(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype), std=1, a=-3, b=3))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]
    
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.weights = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        rms_x = torch.sqrt(reduce(x**2, "... d_model -> ... 1", "mean") + self.eps)
        result = x * self.weights / rms_x
        return result.to(in_dtype)