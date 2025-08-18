import torch

from sparse_but_wrong.toy_models.toy_model import ToyModel


def test_ToyModel_init():
    model = ToyModel(num_feats=10, hidden_dim=10, target_cos_sim=0)
    assert model.embed.weight.data.shape == (10, 10)
    # Check if features are orthogonal
    feature_matrix = model.embed.weight.data.T
    dot_products = feature_matrix @ feature_matrix.T
    assert torch.allclose(
        dot_products, torch.eye(10), atol=1e-6
    ), "Features are not orthogonal"

    # Check if features have norm of 1
    norms = torch.norm(feature_matrix, dim=1)
    assert torch.allclose(
        norms, torch.ones(10), atol=1e-6
    ), "Features do not have unit norm"
