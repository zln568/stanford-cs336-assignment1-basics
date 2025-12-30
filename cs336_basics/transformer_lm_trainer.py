import torch
import math

from jaxtyping import Float, Int
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional

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

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay= 0.1, ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "alpha": lr,
            "betas": betas,
            "eps": eps,
            "wd": weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["alpha"] # Get the learning rate.
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["wd"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data # Get the gradient of loss with respect to p.
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))

                m = beta1 * m + (1-beta1) * grad
                v = beta2 * v + (1-beta2) * grad**2
                a_t = alpha * math.sqrt(1-beta2**t)/(1-beta1**t)
                p.data -= a_t * m / (torch.sqrt(v) + eps) # Update weight tensor in-place.
                p.data *= (1 - alpha * wd)

                state["t"] = t + 1 # Increment iteration number.
                state["m"] = m
                state["v"] = v
        return loss