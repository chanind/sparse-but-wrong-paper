import torch
from sae_lens import SAE

from sparse_but_wrong.toy_models.toy_model import ToyModel


@torch.no_grad()
def init_sae_to_match_model(
    sae: SAE,
    toy_model: ToyModel,
    noise_level: float = 0.0,
    feature_ordering: torch.Tensor | None = None,
) -> None:
    min_dim = min(sae.W_enc.shape[1], toy_model.embed.weight.shape[1])
    features = toy_model.embed.weight[:, :min_dim]
    if feature_ordering is not None:
        features = features[:, feature_ordering]
    sae.W_enc.data[:, :min_dim] = features + torch.randn_like(features) * noise_level
    sae.W_dec.data = sae.W_enc.data.T.clone()
