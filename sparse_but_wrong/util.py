import math
import re
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from functools import cache
from typing import Generic, TypeVar

import torch
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
from tqdm import tqdm

T = TypeVar("T")
DEFAULT_DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_DEVICE = torch.device(DEFAULT_DEVICE_STR)


def cos_sims(mat1: torch.Tensor, mat2: torch.Tensor):
    """
    Calculate the cosine similarity between each row of mat1 and each row of mat2.

    Args:
        mat1: A tensor of shape (n_rows1, n_cols1).
        mat2: A tensor of shape (n_rows2, n_cols2).

    Returns:
        A tensor of shape (n_rows1, n_rows2) containing the cosine similarity between each row of mat1 and each row of mat2.
    """
    mat1_normed = mat1 / (mat1.norm(dim=0, keepdim=True))
    mat2_normed = mat2 / (mat2.norm(dim=0, keepdim=True))

    return mat1_normed.T @ mat2_normed


def dtypify(dtype_str: str) -> torch.dtype:
    return getattr(torch, dtype_str)


def batchify(
    data: Sequence[T], batch_size: int, show_progress: bool = False
) -> Generator[Sequence[T], None, None]:
    """Generate batches from data. If show_progress is True, display a progress bar."""

    for i in tqdm(
        range(0, len(data), batch_size),
        total=(len(data) // batch_size + (len(data) % batch_size != 0)),
        disable=not show_progress,
    ):
        yield data[i : i + batch_size]


def tbatchify(
    data: torch.Tensor, batch_size: int, show_progress: bool = False
) -> Generator[torch.Tensor, None, None]:
    "Wrapper around batchify that handles tensor typing"
    return batchify(data, batch_size, show_progress)  # type: ignore


def untuple_tensor(x: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return x[0] if isinstance(x, tuple) else x


Tweenable = TypeVar("Tweenable", float, list[float])


class Tween(Generic[Tweenable]):
    def __init__(
        self, start: Tweenable, end: Tweenable, n_steps: int, start_step: int = 0
    ):
        self.start = start
        self.end = end
        self.n_steps = n_steps
        self.current = start
        self.start_step = start_step

    def __call__(self, step: int) -> Tweenable:
        if isinstance(self.start, list) and isinstance(self.end, list):
            return [
                _tween_scalar(step, start, end, self.n_steps, self.start_step)
                for start, end in zip(self.start, self.end)
            ]
        else:
            assert isinstance(self.start, float | int) and isinstance(
                self.end, float | int
            )
            return _tween_scalar(
                step, self.start, self.end, self.n_steps, self.start_step
            )

    def is_finished(self, step: int) -> bool:
        return step >= self.start_step + self.n_steps


def _tween_scalar(
    step: int, start: float, end: float, n_steps: int, start_step: int = 0
) -> float:
    if step < start_step:
        return start
    elif step >= start_step + n_steps:
        return end
    else:
        return start + (end - start) * (step - start_step) / n_steps


def listify(x: T | list[T]) -> list[T]:
    return x if isinstance(x, list) else [x]


# Copied from https://github.com/azaitsev/millify/blob/master/millify/__init__.py


def remove_exponent(d):
    """Remove exponent."""
    return d.quantize(Decimal(1)) if d == d.to_integral() else d.normalize()


def millify(n, precision=0, drop_nulls=True, prefixes=[]):
    """Humanize number."""
    millnames = ["", "k", "M", "B", "T", "P", "E", "Z", "Y"]
    if prefixes:
        millnames = [""]
        millnames.extend(prefixes)
    n = float(n)
    millidx = max(
        0,
        min(
            len(millnames) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))
        ),
    )
    result = "{:.{precision}f}".format(n / 10 ** (3 * millidx), precision=precision)
    if drop_nulls:
        result = remove_exponent(Decimal(result))
    return f"{result}{millnames[millidx]}"


class GradientScaler(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, input: torch.Tensor, scale: float
    ):
        ctx.scale = scale  # type: ignore
        return input

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor):  # type: ignore
        scaled_grad = grad_output * ctx.scale  # type: ignore
        return scaled_grad, None


def scale_grad(input: torch.Tensor, scale: float) -> torch.Tensor:
    return GradientScaler.apply(input, scale)  # type: ignore


class ScaleParallelGradients(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        dim: int,
        scale: float,
    ):
        ctx.save_for_backward(input)
        ctx.dim = dim  # type: ignore
        ctx.scale = scale  # type: ignore
        return input

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor):  # type: ignore
        dim: int = ctx.dim  # type: ignore
        scale: float = ctx.scale  # type: ignore
        input: torch.Tensor = ctx.saved_tensors[0]  # type: ignore

        # Calculate the projection of grad_output onto input along the specified dimension
        # proj = (grad_output · input) / (input · input) * input
        dot_product = (grad_output * input).sum(dim=dim, keepdim=True)
        input_squared_norm = (input * input).sum(dim=dim, keepdim=True) + 1e-8
        projection = (dot_product / input_squared_norm) * input

        # Scale the parallel component
        new_grad_output = grad_output - projection * (1.0 - scale)
        return new_grad_output, None, None


def scale_parallel_gradients(
    input: torch.Tensor, dim: int, scale: float
) -> torch.Tensor:
    return ScaleParallelGradients.apply(input, dim, scale)  # type: ignore


@dataclass
class SaeInfo:
    l0: int
    layer: int
    width: int
    path: str
    release: str


@cache
def get_gemmascope_saes_info(
    layer: int | None = None, release: str = "gemma-scope-2b-pt-res"
) -> list[SaeInfo]:
    """
    Get a list of all available Gemmascope SAEs, optionally filtering by a specific layer.
    """
    gemma_2_saes = get_pretrained_saes_directory()[release]
    saes = []
    for sae_name, sae_path in gemma_2_saes.saes_map.items():
        l0 = int(gemma_2_saes.expected_l0[sae_name])
        width_match = re.search(r"width_(\d+)(k|m)", sae_name)
        assert width_match is not None
        assert width_match.group(2) in ["k", "m"]
        width = int(width_match.group(1)) * 1000
        if width_match.group(2) == "m":
            width *= 1000
        layer_match = re.search(r"layer_(\d+)", sae_name)
        # new embedding SAEs don't have a layer; we don't care about them, so just skip
        if layer_match is None:
            continue
        sae_layer = int(layer_match.group(1))
        if layer is None or sae_layer == layer:
            saes.append(
                SaeInfo(
                    l0=l0, layer=sae_layer, width=width, path=sae_path, release=release
                )
            )
    return saes
