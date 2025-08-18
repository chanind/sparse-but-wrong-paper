import pytest
from datasets import Dataset
from sae_lens import SAE
from transformer_lens import HookedTransformer


@pytest.fixture
def gpt2_model():
    return HookedTransformer.from_pretrained("gpt2", device="cpu")


@pytest.fixture
def gpt2_tokenizer(gpt2_model: HookedTransformer):
    return gpt2_model.tokenizer


@pytest.fixture
def gpt2_l4_sae() -> SAE:
    return SAE.from_pretrained(
        "gpt2-small-res-jb", "blocks.4.hook_resid_pre", device="cpu"
    )


@pytest.fixture
def example_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
        * 20
    )
