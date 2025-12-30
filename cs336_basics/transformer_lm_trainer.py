import torch

from jaxtyping import Float, Int
from torch import Tensor

def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    log_probs = log_softmax(inputs)
    target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return -target_log_probs.mean()
    

def log_softmax(inputs: Float[Tensor, " batch_size vocab_size"]):
    max = inputs.amax(dim=-1, keepdim=True)
    shifted = inputs - max
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
    return shifted - log_sum_exp