from collections.abc import Callable
from dataclasses import dataclass

import torch
from sae_lens import SAE

from sparse_but_wrong.toy_models.toy_model import ToyModel


@dataclass
class EvalResult:
    true_l0: float
    sae_l0: float
    dead_features: int
    shrinkage: float


def eval_sae(
    sae: SAE, toy_model: ToyModel, generate_batch: Callable[[int], torch.Tensor]
):
    sae.eval()
    sae.fold_W_dec_norm()
    samples = generate_batch(100_000)
    true_l0 = (samples > 0).float().sum(dim=-1).mean().item()
    true_acts = toy_model.embed(samples)
    sae_features = sae.encode(true_acts)
    sae_output = sae.decode(sae_features)
    sae_l0 = (sae_features > 0).float().sum(dim=-1).mean().item()
    dead_features = (
        ((sae_features == 0).sum(dim=0) == sae_features.shape[0]).sum()
    ).item()
    shrinkage = (sae_output.norm(dim=-1) / true_acts.norm(dim=-1)).mean().item()

    return EvalResult(
        true_l0=true_l0,
        sae_l0=sae_l0,
        shrinkage=shrinkage,
        dead_features=int(dead_features),
    )
