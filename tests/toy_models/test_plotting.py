from unittest.mock import patch

import torch
from sae_lens import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)

from sparse_but_wrong.toy_models.plotting import (
    _find_best_index_reordering,
    plot_sae_feat_cos_sims_seaborn,
)
from sparse_but_wrong.toy_models.toy_model import ToyModel


def test_plot_sae_feat_cos_sims_seaborn_reorder_features_bool():
    """Test that reorder_features=True applies automatic reordering."""
    # Set up dimensions
    num_feats = 3
    num_hidden = 4

    # Create toy model
    model = ToyModel(
        num_feats=num_feats, hidden_dim=num_hidden, target_cos_sim=0, bias=False
    )

    # Create SAE
    cfg = BatchTopKTrainingSAEConfig(d_in=num_hidden, d_sae=num_feats, k=2)
    sae = BatchTopKTrainingSAE(cfg)

    # Mock plt.show to prevent actual plotting during tests
    with patch("matplotlib.pyplot.show"):
        # This should not raise any errors
        plot_sae_feat_cos_sims_seaborn(
            sae=sae, model=model, title_suffix="test", reorder_features=True
        )


def test_plot_sae_feat_cos_sims_seaborn_reorder_features_tensor():
    """Test that reorder_features with tensor applies custom ordering."""
    # Set up dimensions
    num_feats = 3
    num_hidden = 4

    # Create toy model
    model = ToyModel(
        num_feats=num_feats, hidden_dim=num_hidden, target_cos_sim=0, bias=False
    )

    # Create SAE
    cfg = BatchTopKTrainingSAEConfig(d_in=num_hidden, d_sae=num_feats, k=2)
    sae = BatchTopKTrainingSAE(cfg)

    # Custom ordering tensor - reverse order
    custom_ordering = torch.tensor([2, 1, 0])

    # Mock plt.show to prevent actual plotting during tests
    with patch("matplotlib.pyplot.show"):
        # This should not raise any errors
        plot_sae_feat_cos_sims_seaborn(
            sae=sae, model=model, title_suffix="test", reorder_features=custom_ordering
        )


def test_plot_sae_feat_cos_sims_seaborn_no_reorder():
    """Test that reorder_features=False (default) works without reordering."""
    # Set up dimensions
    num_feats = 3
    num_hidden = 4

    # Create toy model
    model = ToyModel(
        num_feats=num_feats, hidden_dim=num_hidden, target_cos_sim=0, bias=False
    )

    # Create SAE
    cfg = BatchTopKTrainingSAEConfig(d_in=num_hidden, d_sae=num_feats, k=2)
    sae = BatchTopKTrainingSAE(cfg)

    # Mock plt.show to prevent actual plotting during tests
    with patch("matplotlib.pyplot.show"):
        # This should not raise any errors
        plot_sae_feat_cos_sims_seaborn(
            sae=sae, model=model, title_suffix="test", reorder_features=False
        )


def test_plot_sae_feat_cos_sims_seaborn_decoder_only():
    """Test that decoder_only=True shows only the decoder plot."""
    # Set up dimensions
    num_feats = 3
    num_hidden = 4

    # Create toy model
    model = ToyModel(
        num_feats=num_feats, hidden_dim=num_hidden, target_cos_sim=0, bias=False
    )

    # Create SAE
    cfg = BatchTopKTrainingSAEConfig(d_in=num_hidden, d_sae=num_feats, k=2)
    sae = BatchTopKTrainingSAE(cfg)

    # Mock plt.show to prevent actual plotting during tests
    with patch("matplotlib.pyplot.show"):
        # This should not raise any errors
        plot_sae_feat_cos_sims_seaborn(
            sae=sae, model=model, title_suffix="test", decoder_only=True
        )


def test_plot_sae_feat_cos_sims_seaborn_decoder_only_with_reordering():
    """Test that decoder_only=True works with reordering."""
    # Set up dimensions
    num_feats = 3
    num_hidden = 4

    # Create toy model
    model = ToyModel(
        num_feats=num_feats, hidden_dim=num_hidden, target_cos_sim=0, bias=False
    )

    # Create SAE
    cfg = BatchTopKTrainingSAEConfig(d_in=num_hidden, d_sae=num_feats, k=2)
    sae = BatchTopKTrainingSAE(cfg)

    # Mock plt.show to prevent actual plotting during tests
    with patch("matplotlib.pyplot.show"):
        # This should not raise any errors
        plot_sae_feat_cos_sims_seaborn(
            sae=sae,
            model=model,
            title_suffix="test",
            decoder_only=True,
            reorder_features=True,
        )


def test_plot_sae_feat_cos_sims_seaborn_dtick():
    """Test that dtick parameter controls tick spacing."""
    # Set up dimensions
    num_feats = 10
    num_hidden = 8

    # Create toy model
    model = ToyModel(
        num_feats=num_feats, hidden_dim=num_hidden, target_cos_sim=0, bias=False
    )

    # Create SAE
    cfg = BatchTopKTrainingSAEConfig(d_in=num_hidden, d_sae=num_feats, k=2)
    sae = BatchTopKTrainingSAE(cfg)

    # Mock plt.show to prevent actual plotting during tests
    with patch("matplotlib.pyplot.show"):
        # Test with dtick=2
        plot_sae_feat_cos_sims_seaborn(
            sae=sae, model=model, title_suffix="test", dtick=2
        )

        # Test with dtick=5
        plot_sae_feat_cos_sims_seaborn(
            sae=sae, model=model, title_suffix="test", dtick=5
        )

        # Test with dtick=None (default behavior)
        plot_sae_feat_cos_sims_seaborn(
            sae=sae, model=model, title_suffix="test", dtick=None
        )


def test_plot_sae_feat_cos_sims_seaborn_dtick_with_one_based_indexing():
    """Test that dtick works with one-based indexing."""
    # Set up dimensions
    num_feats = 8
    num_hidden = 6

    # Create toy model
    model = ToyModel(
        num_feats=num_feats, hidden_dim=num_hidden, target_cos_sim=0, bias=False
    )

    # Create SAE
    cfg = BatchTopKTrainingSAEConfig(d_in=num_hidden, d_sae=num_feats, k=2)
    sae = BatchTopKTrainingSAE(cfg)

    # Mock plt.show to prevent actual plotting during tests
    with patch("matplotlib.pyplot.show"):
        # Test with dtick=3 and one-based indexing
        plot_sae_feat_cos_sims_seaborn(
            sae=sae,
            model=model,
            title_suffix="test",
            dtick=3,
            one_based_indexing=True,
        )


def test_plot_sae_feat_cos_sims_seaborn_dtick_decoder_only():
    """Test that dtick works with decoder_only mode."""
    # Set up dimensions
    num_feats = 6
    num_hidden = 4

    # Create toy model
    model = ToyModel(
        num_feats=num_feats, hidden_dim=num_hidden, target_cos_sim=0, bias=False
    )

    # Create SAE
    cfg = BatchTopKTrainingSAEConfig(d_in=num_hidden, d_sae=num_feats, k=2)
    sae = BatchTopKTrainingSAE(cfg)

    # Mock plt.show to prevent actual plotting during tests
    with patch("matplotlib.pyplot.show"):
        # Test with dtick=2 and decoder_only
        plot_sae_feat_cos_sims_seaborn(
            sae=sae,
            model=model,
            title_suffix="test",
            dtick=2,
            decoder_only=True,
        )


def test_find_best_index_reordering():
    """Test the _find_best_index_reordering function directly."""
    # Create a simple cos_sims matrix where reordering would improve alignment
    cos_sims = torch.tensor(
        [
            [0.1, 0.9, 0.2],  # latent 0 best matches feature 1
            [0.8, 0.1, 0.3],  # latent 1 best matches feature 0
            [0.2, 0.3, 0.7],  # latent 2 best matches feature 2
        ]
    )

    score, sorted_indices = _find_best_index_reordering(cos_sims)

    # Check that we get a score and indices
    assert isinstance(score, float)
    assert isinstance(sorted_indices, torch.Tensor)
    assert sorted_indices.shape == (3,)

    # Check that all indices are present (permutation)
    assert set(sorted_indices.tolist()) == {0, 1, 2}
