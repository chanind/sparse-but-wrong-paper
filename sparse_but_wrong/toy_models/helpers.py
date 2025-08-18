from typing import Any, TypeVar

from sae_lens.training.training_sae import TrainingSAEConfig

from sparse_but_wrong.saes.base_sae import BaseSAERunnerConfig
from sparse_but_wrong.util import DEFAULT_DEVICE

GenericRunnerConfig = TypeVar("GenericRunnerConfig", bound=BaseSAERunnerConfig)
GenericTrainingSAEConfig = TypeVar("GenericTrainingSAEConfig", bound=TrainingSAEConfig)

base_overrides = dict(
    context_size=1000,
    device=str(DEFAULT_DEVICE),
    training_tokens=100_000_000,
    eval_every_n_wandb_logs=99999999999,
    l1_coefficient=5e-3,
    normalize_sae_decoder=False,
    scale_sparsity_penalty_by_decoder_norm=True,
    lr=3e-4,
    log_to_wandb=False,
    apply_b_dec_to_input=True,
    b_dec_init_method="zeros",
)


def quick_cfgs(
    overrides: dict[str, Any],
    runner_config_class: type[GenericRunnerConfig] = BaseSAERunnerConfig,
    training_sae_config_class: type[GenericTrainingSAEConfig] = TrainingSAEConfig,
) -> tuple[GenericRunnerConfig, GenericTrainingSAEConfig]:
    """
    Helper to create a runner config and training SAE config without having to set every default for toy models.
    """
    runner_cfg = runner_config_class(**{**base_overrides, **overrides})
    sae_cfg: GenericTrainingSAEConfig = (
        training_sae_config_class.from_sae_runner_config(runner_cfg)
    )  # type: ignore
    return runner_cfg, sae_cfg
