# Sparse but Wrong

[![build](https://github.com/chanind/sparse-but-wrong-paper/actions/workflows/ci.yaml/badge.svg)](https://github.com/chanind/sparse-but-wrong-paper/actions/workflows/ci.yaml)

This repo contains the code for the paper [Sparse but Wrong: Incorrect L0 Leads to Incorrect
Features in Sparse Autoencoders](https://arxiv.org/abs/2508.16560).

## Setup

This project uses uv for package management. You can install dependencies with:

```bash
uv sync
```

Or, you can just use pip and run:

```bash
pip install -e .
```

## Running the experiments

The toy model experiments are all in the `notebooks` directory, with supporting files in the `toy_model` directory. We also provide a demo notebook for how to reproduce our Gemma-2-2b experiments in `notebooks/train_and_eval_llm_sae.ipynb`.

We extend the SAELens BatchTopKSAE class in `enhanced_batch_topk_sae.py` to keep the decoder normalized and support our experiments where we change the L0 of the SAE during training.

## development

We use `pytest` for testing. You can run the tests with:

```bash
uv run pytest
```

We also use `ruff` for linting and `pyright` for type checking. You can run the linting and type checking with:

```bash
uv run ruff check .
uv run pyright
```

## Citation

```
@article{chanin2025sparse,
     title={Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders},
     author={David Chanin and Adrià Garriga-Alonso},
     year={2025},
     journal={arXiv preprint arXiv:2508.16560}
}
```
