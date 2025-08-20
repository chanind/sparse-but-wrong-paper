from dataclasses import dataclass

import torch
from sae_lens import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
from sae_lens.saes.sae import TrainStepInput, TrainStepOutput
from typing_extensions import override

from sparse_but_wrong.util import Tween


@dataclass
class EnhancedBatchTopKTrainingSAEConfig(BatchTopKTrainingSAEConfig):
    """
    Configuration class for training a EnhancedBatchTopKTrainingSAE.

    Contains the following extra parameters:
    - normalize_acts_by_decoder_norm: Whether to scale acts during training as if the decoder is normalized.
    - initial_k: Initial k value, if we want to increase or decrease the k during training.
    - transition_k_duration_steps: The number of steps over which to transition from the initial k to the final k.
    - transition_k_start_step: The step at which to start the transition.
    """

    normalize_acts_by_decoder_norm: bool = False
    initial_k: int | None = None
    transition_k_duration_steps: int | None = None
    transition_k_start_step: int = 0


class EnchancedBatchTopkTrainingSAE(BatchTopKTrainingSAE):
    """
    BatchTopK variant with some extra functionality.
    """

    cfg: EnhancedBatchTopKTrainingSAEConfig  # type: ignore[assignment]
    transition_k_tween: Tween | None = None
    step_num: int = 0

    def __init__(
        self, cfg: EnhancedBatchTopKTrainingSAEConfig, use_error_term: bool = False
    ):
        super().__init__(cfg, use_error_term)
        if cfg.initial_k is not None and cfg.transition_k_duration_steps is not None:
            self.transition_k_tween = Tween(
                start=cfg.initial_k,
                end=cfg.k,
                n_steps=cfg.transition_k_duration_steps,
                start_step=cfg.transition_k_start_step,
            )
        self._update_k()

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        output = super().training_forward_pass(step_input)
        self.step_num += 1
        self._update_k()
        return output

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode, while scaling the hidden pre by the decoder norm if we're keeping decoder normalized.
        """
        sae_in = self.process_sae_in(x)

        # need to scale the hidden pre by the decoder norm if we're keeping decoder normalized
        W_enc = self.W_enc
        if self.cfg.normalize_acts_by_decoder_norm:
            W_enc = W_enc * self.W_dec.norm(dim=-1)

        hidden_pre = self.hook_sae_acts_pre(sae_in @ W_enc + self.b_enc)

        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))
        return feature_acts, hidden_pre

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Decode, while normalizing the feature acts by the decoder norm if we're keeping decoder normalized.
        """
        if self.cfg.normalize_acts_by_decoder_norm:
            feature_acts = feature_acts / self.W_dec.norm(dim=-1)
        return super().decode(feature_acts)

    @override
    @torch.no_grad()
    def fold_W_dec_norm(self):
        # If we're keeping decoder normalized, we can fold the norm without issue, since we're already acting as if the decoder is normalized
        if self.cfg.normalize_acts_by_decoder_norm:
            W_dec_norms = self.W_dec.norm(dim=-1).unsqueeze(1)
            self.W_dec.data = self.W_dec.data / W_dec_norms
            self.W_enc.data = self.W_enc.data * W_dec_norms.T
        else:
            super().fold_W_dec_norm()

    def _update_k(self):
        if self.transition_k_tween is None:
            return
        k = self.transition_k_tween(self.step_num)
        self.activation_fn.k = int(k)  # type: ignore[attr-defined]
