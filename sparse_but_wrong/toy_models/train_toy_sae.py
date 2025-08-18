from collections.abc import Callable

import torch
from sae_lens import LoggingConfig, TrainingSAE
from sae_lens.training.sae_trainer import SAETrainer, SAETrainerConfig
from tqdm import tqdm

from sparse_but_wrong.toy_models.toy_model import ToyModel
from sparse_but_wrong.util import DEFAULT_DEVICE


# We just need something that's an iterator over training activations
class DataIterator:
    def __init__(self, model, generate_batch_fn, batch_size: int):
        self.model = model
        self.batch_size = batch_size
        self.generate_batch_fn = generate_batch_fn

    @torch.no_grad()
    def next_batch(self):
        return self.model(self.generate_batch_fn(self.batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()


def train_toy_sae(
    sae: TrainingSAE,
    toy_model: ToyModel,
    activations_batch_provider: Callable[[int], torch.Tensor],
    lr: float = 3e-4,
    training_tokens: int = 15_000_000,
    lr_warm_up_steps: int = 0,
    lr_decay_steps: int = 0,
    device: torch.device = DEFAULT_DEVICE,
    train_batch_size_tokens: int = 1024,
) -> None:
    tqdm._instances.clear()  # type: ignore
    training_cfg = SAETrainerConfig(
        n_checkpoints=0,
        checkpoint_path="checkpoints",
        total_training_samples=training_tokens,
        device=str(device),
        autocast=False,
        lr=lr,
        lr_end=lr,
        lr_scheduler_name="constant",
        lr_warm_up_steps=lr_warm_up_steps,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr_decay_steps=lr_decay_steps,
        n_restart_cycles=1,
        train_batch_size_samples=train_batch_size_tokens,
        dead_feature_window=1000,
        feature_sampling_window=2000,
        logger=LoggingConfig(log_to_wandb=False),
    )
    toy_model.eval()
    data_iterator = DataIterator(
        toy_model, activations_batch_provider, train_batch_size_tokens
    )
    trainer = SAETrainer(
        data_provider=data_iterator,
        sae=sae,
        cfg=training_cfg,
    )
    trainer.fit()
