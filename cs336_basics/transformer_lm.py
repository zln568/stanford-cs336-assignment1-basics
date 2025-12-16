import torch
import numpy as np

from einops import einsum, reduce, rearrange
from jaxtyping import Float, Bool, Int
from torch import Tensor

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
    
def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    w1_x = einsum(w1_weight, in_features, "d_ff d_model, ... d_model -> ... d_ff")
    silu = w1_x * torch.sigmoid(w1_x)

    w3_x = einsum(w3_weight, in_features, "d_ff d_model, ... d_model -> ... d_ff")
    return einsum(w2_weight, silu * w3_x, "d_model d_ff, ... d_ff -> ... d_model")

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        i = torch.arange(max_seq_len, device=device).unsqueeze(1)
        pow = torch.arange(0, d_k, 2, device=device) / d_k
        angles = i / (theta**pow)

        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_even_rot = cos * x_even - sin * x_odd
        x_odd_rot = sin * x_even + cos * x_odd

        x_rot = rearrange([x_even_rot, x_odd_rot], "two ... -> ... two")
        result = rearrange(x_rot, "... d1 d2 -> ... (d1 d2)")
        return result
    
def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    max = in_features.amax(dim=dim, keepdim=True)
    x_exp = torch.exp(in_features - max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    pre_softmax_value = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / np.sqrt(d_k)
    if mask is not None:
        pre_softmax_value = torch.where(mask == True, pre_softmax_value, -torch.inf)
    
    return einsum(run_softmax(pre_softmax_value, -1), V, "... queries values, ... values d_v -> ... queries d_v")

class CausalMultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.o_proj = Linear(d_model, d_model, device, dtype)

        self.num_heads = num_heads

    def forward(self, 
                in_features: Float[Tensor, " ... sequence_length d_in"],
                rope: RotaryPositionalEmbedding | None = None,
                token_positions: Int[Tensor, " ... sequence_length"] | None = None):
        q = self.q_proj.forward(in_features)
        k = self.k_proj.forward(in_features)
        v = self.v_proj.forward(in_features)

        q = rearrange(q, "... s (h d) -> ... h s d", h=self.num_heads)
        k = rearrange(k, "... s (h d) -> ... h s d", h=self.num_heads)
        v = rearrange(v, "... s (h d) -> ... h s d", h=self.num_heads)

        if rope is not None:
            q = rope.forward(q, token_positions)
            k = rope.forward(k, token_positions)

        mask = ~torch.triu(torch.full((q.shape[-2], k.shape[-2]), True, device=in_features.device), diagonal=1)

        multihead = run_scaled_dot_product_attention(q, k, v, mask)
        multihead = rearrange(multihead, "... h s d -> ... s (h d)")

        return self.o_proj(multihead)