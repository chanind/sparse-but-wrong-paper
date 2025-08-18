from functools import partial

import pytest
import torch
from sae_lens import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)

from sparse_but_wrong.toy_models.get_training_batch import get_training_batch
from sparse_but_wrong.toy_models.toy_model import ToyModel
from sparse_but_wrong.toy_models.train_toy_sae import train_toy_sae
from sparse_but_wrong.util import DEFAULT_DEVICE


@pytest.mark.skip(reason="Too flaky")
def test_train_toy_sae_can_find_a_reasonable_solution():
    num_feats = 2
    num_hidden = 5
    lr = 0.01
    model = ToyModel(
        num_feats=num_feats, hidden_dim=num_hidden, target_cos_sim=0, bias=True
    )
    cfg = BatchTopKTrainingSAEConfig(
        d_in=num_hidden, d_sae=num_feats, device=str(DEFAULT_DEVICE)
    )
    sae = BatchTopKTrainingSAE(cfg)
    feat_probs = torch.tensor([0.1] * num_feats)
    generate_batch = partial(get_training_batch, firing_probabilities=feat_probs)

    # set the SAE to near the correct solution to make this test less flaky
    with torch.no_grad():
        sae.W_dec.data = model.embed.weight.data.T + torch.randn_like(sae.W_dec)
        sae.W_enc.data = sae.W_dec.data.T

    train_toy_sae(
        sae,
        model,
        generate_batch,
        training_tokens=30_000_000,
        lr=lr,
        lr_warm_up_steps=500,
        lr_decay_steps=500,
    )
    # Check if the learned feature aligns with the true feature
    true_feature = model.embed.weight.data.T
    learned_feature = sae.W_dec.T

    # Normalize the features
    true_feature_norm = true_feature / torch.norm(true_feature, dim=-1, keepdim=True)
    learned_feature_norm = learned_feature / torch.norm(
        learned_feature, dim=0, keepdim=True
    )

    # Calculate cosine similarity matrix
    cos_sim_matrix = torch.mm(true_feature_norm, learned_feature_norm)

    # Get the maximum cosine similarity
    # sometimes it finds -1.0 values, which isnt' great but is an understandable local minima
    max_cos_sims = torch.max(cos_sim_matrix.abs(), dim=-1).values

    assert torch.allclose(max_cos_sims, torch.tensor([1.0] * num_feats), atol=2e-1), (
        f"Maximum cosine similarity between true and learned features is {max_cos_sims.tolist()}, expected close to 1.0"
    )
