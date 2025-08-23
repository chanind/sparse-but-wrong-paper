import torch
from sae_lens import SAE


def nth_decoder_projection(input_acts: torch.Tensor, sae: SAE, n: int) -> torch.Tensor:
    """
    Calculate the nth decoder projection of the SAE for the given input acts.

    Args:
        input_acts: The input acts, shape (batch_size, d_in).
        sae: The SAE to evaluate.
        n: The nth decoder projection to calculate.

    Returns:
        The nth decoder projection.
    """
    hidden_pre_dec = (input_acts - sae.b_dec) @ sae.W_dec.T
    sorted_hidden_pre_dec = hidden_pre_dec.flatten().sort(descending=True).values
    index = n * hidden_pre_dec.shape[0]
    return sorted_hidden_pre_dec[index]
