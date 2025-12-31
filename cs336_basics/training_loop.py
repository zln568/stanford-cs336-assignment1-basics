import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start_idx = dataset.size - context_length - 1
    start_indices = np.random.randint(0, max_start_idx+1, size=batch_size)

    x_batches = np.zeros((batch_size, context_length))
    y_batches = np.zeros((batch_size, context_length))
    for i, idx in enumerate(start_indices):
        x_batches[i] = dataset[idx : idx+context_length]
        y_batches[i] = dataset[idx+1 : idx+context_length+1]

    x_batches = torch.from_numpy(x_batches).to(device)
    y_batches = torch.from_numpy(y_batches).to(device)
    
    return (x_batches, y_batches)