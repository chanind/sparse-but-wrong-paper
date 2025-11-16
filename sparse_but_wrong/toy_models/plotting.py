from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import torch
from plotly.subplots import make_subplots
from sae_lens import TrainingSAE
from tueplots import axes, bundles

from sparse_but_wrong.toy_models.toy_model import ToyModel
from sparse_but_wrong.util import cos_sims


def _find_best_index_reordering(
    cos_sims: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """Find the best index reordering of the cos_sims tensor.

    Args:
        cos_sims: A tensor of cosine similarities between two sets of vectors.

    Returns:
        the score of the reordering, and the reordered tensor
    """
    best_feature_matches = torch.argmax(torch.abs(cos_sims), dim=1)
    # Sort SAE latents by their best matching true feature
    sorted_indices = torch.argsort(best_feature_matches)
    score = cos_sims[sorted_indices, torch.arange(cos_sims.shape[1])].mean().item()
    return score, sorted_indices


def find_best_index_reordering_for_saes(
    saes: Iterable[TrainingSAE],
    model: ToyModel,
) -> torch.Tensor:
    best_score = float("-inf")
    best_ordering = None
    for sae in saes:
        dec_cos_sims = (
            torch.round(cos_sims(sae.W_dec.T, model.embed.weight) * 100) / 100 + 0.0
        )
        score, ordering = _find_best_index_reordering(dec_cos_sims)
        if score > best_score:
            best_score = score
            best_ordering = ordering
    if best_ordering is None:
        raise ValueError("No best ordering found")
    return best_ordering


def plot_sae_feat_cos_sims(
    sae: TrainingSAE,
    model: ToyModel,
    title_suffix: str,
    height: int = 400,
    width: int = 800,
    show_values: bool = False,  # New parameter to control showing values
    reorder_features: bool | torch.Tensor = False,
    show_plot: bool = True,
    save_path: str | Path | None = None,
    dtick: int | None = 1,
    scale: float = 1.0,  # Scale factor for image resolution
):
    dec_cos_sims = (
        torch.round(cos_sims(sae.W_dec.T, model.embed.weight) * 100) / 100 + 0.0
    )
    enc_cos_sims = (
        torch.round(cos_sims(sae.W_enc, model.embed.weight) * 100) / 100 + 0.0
    )

    if reorder_features is not False:
        if isinstance(reorder_features, bool):
            _, sorted_indices = _find_best_index_reordering(dec_cos_sims)
            dec_cos_sims = dec_cos_sims[sorted_indices]
            enc_cos_sims = enc_cos_sims[sorted_indices]
        else:
            dec_cos_sims = dec_cos_sims[reorder_features]
            enc_cos_sims = enc_cos_sims[reorder_features]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("SAE encoder", "SAE decoder"))
    hovertemplate = "True feature: %{x}<br>SAE Latent: %{y}<br>Cosine Similarity: %{z:.3f}<extra></extra>"

    # Create encoder heatmap trace with conditional text properties
    encoder_args = {
        "z": enc_cos_sims.detach().cpu().numpy(),
        "zmin": -1,
        "zmax": 1,
        "colorscale": "RdBu",
        "showscale": False,
        "hovertemplate": hovertemplate,
    }

    # Only add text-related properties if show_values is True
    if show_values:
        encoder_args["texttemplate"] = "%{z:.2f}"
        encoder_args["textfont"] = {"size": 10}

    fig.add_trace(go.Heatmap(**encoder_args), row=1, col=1)

    # Create decoder heatmap trace with conditional text properties
    decoder_args = {
        "z": dec_cos_sims.detach().cpu().numpy(),
        "zmin": -1,
        "zmax": 1,
        "colorscale": "RdBu",
        "colorbar": dict(title="cos sim", x=1.0, dtick=1, tickvals=[-1, 0, 1]),
        "hovertemplate": hovertemplate,
    }

    # Only add text-related properties if show_values is True
    if show_values:
        decoder_args["texttemplate"] = "%{z:.2f}"
        decoder_args["textfont"] = {"size": 10}

    fig.add_trace(go.Heatmap(**decoder_args), row=1, col=2)

    fig.update_layout(
        height=height,
        width=width,
        title_text=f"Cosine Similarity with True Features ({title_suffix})",
    )
    fig.update_xaxes(title_text="True feature", row=1, col=1, dtick=dtick)
    fig.update_xaxes(title_text="True feature", row=1, col=2, dtick=dtick)
    fig.update_yaxes(title_text="SAE Latent", row=1, col=1, dtick=dtick)
    fig.update_yaxes(title_text="SAE Latent", row=1, col=2, dtick=dtick)

    if show_plot:
        fig.show()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(save_path, scale=scale)


SEABORN_RC_CONTEXT = {
    **bundles.neurips2021(),
    **axes.lines(),
    # Use constrained_layout for automatic adjustments
    "figure.constrained_layout.use": True,
}


def plot_sae_feat_cos_sims_seaborn(
    sae: TrainingSAE,
    model: ToyModel,
    title_suffix: str | None = None,
    title: str | None = None,
    height: float = 8,
    width: float = 16,
    show_values: bool = False,
    save_path: str | Path | None = None,
    one_based_indexing: bool = False,
    reorder_features: bool | torch.Tensor = False,
    decoder_only: bool = False,
    dtick: int | None = 1,
    decoder_title: str | None = "SAE decoder",
    adjust_for_superposition: bool = False,
) -> None:
    """Plot cosine similarities between SAE features and true features using seaborn.

    Args:
        sae: The trained SAE
        model: The toy model being analyzed
        title_suffix: Suffix to add to the plot title
        title: Custom title for the plot
        height: Figure height in inches
        width: Figure width in inches
        show_values: Whether to show the cosine similarity values on the heatmap
        save_path: Optional path to save the figure
        one_based_indexing: Whether to use 1-based indexing for axis labels
        reorder_features: Whether to reorder features for better visualization.
            If True, automatically finds best ordering. If a tensor, uses that ordering.
        decoder_only: Whether to plot only the decoder (single plot instead of side-by-side)
        dtick: Spacing between ticks on both axes. If None, uses matplotlib's default spacing.
        adjust_for_superposition: Whether to subtract out superposition noise from the cosine similarities.
    """
    dec_cos_sims = (
        torch.round(cos_sims(sae.W_dec.T, model.embed.weight) * 100) / 100 + 0.0
    )
    enc_cos_sims = (
        torch.round(cos_sims(sae.W_enc, model.embed.weight) * 100) / 100 + 0.0
    )

    # Apply feature reordering if requested
    if reorder_features is not False:
        if isinstance(reorder_features, bool):
            _, sorted_indices = _find_best_index_reordering(dec_cos_sims)
            dec_cos_sims = dec_cos_sims[sorted_indices]
            enc_cos_sims = enc_cos_sims[sorted_indices]
        else:
            dec_cos_sims = dec_cos_sims[reorder_features]
            enc_cos_sims = enc_cos_sims[reorder_features]

    if adjust_for_superposition:
        feature_cos_sims = cos_sims(model.embed.weight, model.embed.weight)
        # Zero out the diagonal to leave it untouched
        feature_cos_sims = feature_cos_sims - torch.diag(torch.diag(feature_cos_sims))
        dec_cos_sims = dec_cos_sims - feature_cos_sims
        enc_cos_sims = enc_cos_sims - feature_cos_sims

    # NOTE: We plot the original matrices, not flipped ones.
    # We will invert the y-axis later for correct visual orientation.

    plt.rcParams.update({"figure.dpi": 150})
    with plt.rc_context(SEABORN_RC_CONTEXT):
        # Create figure and subplots based on decoder_only setting
        if decoder_only:
            fig, ax = plt.subplots(1, 1, figsize=(width / 2, height))
            ax2 = ax  # Use single axis as decoder axis
            ax1 = None  # No encoder axis
        else:
            fig, axes = plt.subplots(1, 2, figsize=(width, height))
            ax1, ax2 = axes

        # Get dimensions for tick labels
        n_features = model.embed.weight.shape[1]
        n_latents = sae.W_enc.shape[1]

        # Create tick labels based on indexing preference and dtick spacing
        if dtick is not None:
            # Create ticks with specified spacing
            if one_based_indexing:
                feature_tick_positions = list(range(1, n_features + 1, dtick))
                latent_tick_positions = list(range(1, n_latents + 1, dtick))
            else:
                feature_tick_positions = list(range(0, n_features, dtick))
                latent_tick_positions = list(range(0, n_latents, dtick))

            # Convert to strings for matplotlib
            feature_ticks = [str(i) for i in feature_tick_positions]
            latent_ticks = [str(i) for i in latent_tick_positions]

            # Calculate actual tick positions for matplotlib (0-based, centered on cells)
            feature_tick_locs = [
                i - (1 if one_based_indexing else 0) + 0.5
                for i in feature_tick_positions
            ]
            latent_tick_locs = [
                i - (1 if one_based_indexing else 0) + 0.5
                for i in latent_tick_positions
            ]
        else:
            # Use all ticks (original behavior)
            raw_feature_ticks = (
                list(range(1, n_features + 1))
                if one_based_indexing
                else list(range(n_features))
            )
            raw_latent_ticks = (
                list(range(1, n_latents + 1))
                if one_based_indexing
                else list(range(n_latents))
            )

            feature_ticks = [str(i) for i in raw_feature_ticks]
            latent_ticks = [str(i) for i in raw_latent_ticks]

            # All tick positions
            feature_tick_locs = [i + 0.5 for i in range(n_features)]
            latent_tick_locs = [i + 0.5 for i in range(n_latents)]

        # Plot encoder heatmap only if not decoder_only
        if not decoder_only and ax1 is not None:
            sns.heatmap(
                enc_cos_sims.detach().cpu().numpy(),  # Use original data
                ax=ax1,
                vmin=-1,
                vmax=1,
                cmap="RdBu",
                center=0,
                annot=show_values,
                fmt=".2f" if show_values else "",
                cbar=False,  # Colorbar handled separately
            )
            ax1.set_title("SAE encoder")
            ax1.set_xlabel("True feature")
            ax1.set_ylabel("SAE Latent")
            ax1.set_xticks(feature_tick_locs, feature_ticks)
            ax1.set_yticks(latent_tick_locs, latent_ticks)
            # Invert y-axis on encoder plot
            ax1.invert_yaxis()

        # Plot decoder heatmap with original data
        sns.heatmap(
            dec_cos_sims.detach().cpu().numpy(),  # Use original data
            ax=ax2,
            vmin=-1,
            vmax=1,
            cmap="RdBu",
            center=0,
            annot=show_values,
            fmt=".2f" if show_values else "",
            cbar=True,  # Add colorbar here
            cbar_kws={
                "label": "cos sim",
                "ticks": [-1, 0, 1],
                "shrink": 0.75,
            },  # Adjust shrink as needed
        )
        if decoder_title is not None:
            ax2.set_title(decoder_title)
        ax2.set_xlabel("True feature")
        ax2.set_ylabel("SAE Latent")  # Always show y-label for decoder
        ax2.set_xticks(feature_tick_locs, feature_ticks)
        ax2.set_yticks(latent_tick_locs, latent_ticks)

        # Invert y-axis on decoder plot (encoder already inverted above if plotted)
        ax2.invert_yaxis()

        # Set the main title
        if title is None:
            if decoder_only:
                title = (
                    f"Decoder cosine similarity with true features ({title_suffix})"
                    if title_suffix
                    else "Decoder cosine similarity with true features"
                )
            else:
                title = (
                    f"Cosine similarity with true features ({title_suffix})"
                    if title_suffix
                    else "Cosine similarity with true features"
                )
        fig.suptitle(title)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                save_path, bbox_inches="tight"
            )  # Use bbox_inches='tight' for saving
        plt.show()


def plot_correlation_matrix(
    correlation_matrix: torch.Tensor,
    save_path: str | Path | None = None,
    dtick: int = 1,
    title: str = "Feature correlation matrix",
    size: tuple[float, float] = (2.5, 2),
) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"figure.dpi": 150})
    with plt.rc_context(SEABORN_RC_CONTEXT):
        plt.figure(figsize=size)
        sns.heatmap(correlation_matrix, cmap="RdBu", center=0, vmin=-1, vmax=1)
        plt.gca().invert_yaxis()
        plt.xlabel("Feature")
        plt.ylabel("Feature")
        plt.title(title)
        # Increase tick spacing to prevent overlapping
        plt.xticks(
            range(0, len(correlation_matrix), dtick),
            [str(i) for i in range(0, len(correlation_matrix), dtick)],
        )
        plt.yticks(
            range(0, len(correlation_matrix), dtick),
            [str(i) for i in range(0, len(correlation_matrix), dtick)],
        )
        # plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
