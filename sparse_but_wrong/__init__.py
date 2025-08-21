from sae_lens import register_sae_training_class

from sparse_but_wrong.enchanced_batch_topk_sae import (
    EnchancedBatchTopKTrainingSAE,
    EnhancedBatchTopKTrainingSAEConfig,
)
from sparse_but_wrong.nth_decoder_projection import nth_decoder_projection
from sparse_but_wrong.toy_models.get_training_batch import (
    generate_random_correlation_matrix,
    get_training_batch,
)
from sparse_but_wrong.toy_models.plotting import (
    plot_sae_feat_cos_sims,
    plot_sae_feat_cos_sims_seaborn,
)
from sparse_but_wrong.toy_models.toy_model import ToyModel
from sparse_but_wrong.toy_models.train_toy_sae import train_toy_sae

register_sae_training_class(
    "enhanced_batchtopk",
    EnchancedBatchTopKTrainingSAE,
    EnhancedBatchTopKTrainingSAEConfig,
)

__all__ = [
    "EnchancedBatchTopKTrainingSAE",
    "EnhancedBatchTopKTrainingSAEConfig",
    "nth_decoder_projection",
    "train_toy_sae",
    "plot_sae_feat_cos_sims",
    "plot_sae_feat_cos_sims_seaborn",
    "get_training_batch",
    "generate_random_correlation_matrix",
    "ToyModel",
]
