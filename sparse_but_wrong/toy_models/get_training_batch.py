import random
from collections.abc import Callable

import torch
from scipy.stats import norm
from torch.distributions import MultivariateNormal

from sparse_but_wrong.util import DEFAULT_DEVICE

######################################################################


def get_correlated_features(
    batch_size: int,
    firing_probabilities: torch.Tensor,
    correlation_matrix: torch.Tensor,
    device: torch.device = DEFAULT_DEVICE,
) -> torch.Tensor:
    """
    Generate correlated binary features using multivariate Gaussian sampling with thresholds.

    Args:
        batch_size: Number of samples to generate
        firing_probabilities: Marginal probabilities for each feature (shape: [num_features])
        correlation_matrix: Correlation matrix between features (shape: [num_features, num_features])
        device: Device to generate samples on

    Returns:
        Binary feature matrix of shape [batch_size, num_features]
    """
    num_features = firing_probabilities.shape[0]

    # Convert marginal probabilities to thresholds using inverse normal CDF
    # We need to use scipy for the inverse normal CDF as PyTorch doesn't have it
    thresholds = torch.tensor(
        [norm.ppf(1 - p.item()) for p in firing_probabilities], device=device
    )

    # Create multivariate normal distribution with correlation matrix as covariance
    mvn = MultivariateNormal(
        loc=torch.zeros(num_features, device=device),
        covariance_matrix=correlation_matrix.to(device),
    )

    # Sample from multivariate normal
    gaussian_samples = mvn.sample((batch_size,))  # [batch_size, num_features]

    # Apply thresholds to get binary features
    # Feature fires if gaussian sample > threshold
    binary_features = (gaussian_samples > thresholds.unsqueeze(0)).float()

    return binary_features


def get_training_batch(
    batch_size: int,
    firing_probabilities: torch.Tensor,  # these are the independent probabilities of each feature firing
    std_firing_magnitudes: (
        torch.Tensor | None
    ) = None,  # If not provided, the stdev of magnitudes will be 0
    mean_firing_magnitudes: (
        torch.Tensor | None
    ) = None,  # If not provided, mean will be 1.0
    device: torch.device = DEFAULT_DEVICE,
    modify_firing_features: Callable[[torch.Tensor], torch.Tensor] | None = None,
    correlation_matrix: (
        torch.Tensor | None
    ) = None,  # If provided, use correlated sampling
):
    if correlation_matrix is not None:
        # Use correlated feature generation
        firing_features = get_correlated_features(
            batch_size, firing_probabilities, correlation_matrix, device
        )
    else:
        # Use independent Bernoulli sampling (original behavior)
        firing_features = torch.bernoulli(
            firing_probabilities.unsqueeze(0).expand(batch_size, -1).to(device)
        )

    if std_firing_magnitudes is None:
        std_firing_magnitudes = torch.zeros_like(firing_probabilities)
    if mean_firing_magnitudes is None:
        mean_firing_magnitudes = torch.ones_like(firing_probabilities)
    mean_firing_magnitudes = mean_firing_magnitudes.to(device)
    if modify_firing_features is not None:
        firing_features = modify_firing_features(firing_features)
    firing_features = firing_features.to(device)
    firing_magnitude_delta = torch.normal(
        torch.zeros_like(firing_probabilities)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .to(device),
        std_firing_magnitudes.unsqueeze(0).expand(batch_size, -1).to(device),
    )
    firing_magnitude_delta[firing_features == 0] = 0
    return (firing_features * (mean_firing_magnitudes + firing_magnitude_delta)).relu()


def _fix_correlation_matrix(
    matrix: torch.Tensor, min_eigenval: float = 1e-6
) -> torch.Tensor:
    """
    Fix a correlation matrix to be positive semi-definite by clipping eigenvalues.

    Args:
        matrix: Input correlation matrix
        min_eigenval: Minimum eigenvalue to ensure (small positive value)

    Returns:
        Fixed positive semi-definite correlation matrix
    """
    # Eigenvalue decomposition
    eigenvals, eigenvecs = torch.linalg.eigh(matrix)

    # Clip negative eigenvalues to small positive value
    eigenvals = torch.clamp(eigenvals, min=min_eigenval)

    # Reconstruct matrix
    fixed_matrix = eigenvecs @ torch.diag(eigenvals) @ eigenvecs.T

    # Ensure diagonal is exactly 1 (correlation matrix property)
    diag_vals = torch.diag(fixed_matrix)
    fixed_matrix = fixed_matrix / torch.sqrt(
        diag_vals.unsqueeze(0) * diag_vals.unsqueeze(1)
    )
    fixed_matrix.fill_diagonal_(1.0)

    return fixed_matrix


def create_correlation_matrix(
    num_features: int,
    correlations: dict[tuple[int, int], float] | None = None,
    default_correlation: float = 0.0,
) -> torch.Tensor:
    """
    Helper function to create correlation matrices.

    Args:
        num_features: Number of features
        correlations: Dict mapping (i, j) pairs to correlation values
        default_correlation: Default correlation for unspecified pairs

    Returns:
        Correlation matrix of shape [num_features, num_features]

    Example:
        # Create correlation matrix with feature 0 and 1 correlated at 0.8
        corr_matrix = create_correlation_matrix(
            num_features=4,
            correlations={(0, 1): 0.8, (2, 3): -0.5}
        )
    """
    matrix = torch.eye(num_features) + default_correlation * (
        1 - torch.eye(num_features)
    )

    if correlations is not None:
        for (i, j), corr in correlations.items():
            matrix[i, j] = corr
            matrix[j, i] = corr  # Ensure symmetry

    # Verify positive definiteness (correlation matrix must be PSD)
    eigenvals = torch.linalg.eigvals(matrix)
    if torch.any(eigenvals.real < -1e-6):
        print("Warning: Correlation matrix is not positive semi-definite!")
        print(f"Minimum eigenvalue: {eigenvals.real.min()}")
        print("Fixing matrix to be positive semi-definite...")
        matrix = _fix_correlation_matrix(matrix)

    return matrix


def generate_random_correlation_matrix(
    num_features: int,
    positive_ratio: float = 0.5,
    correlation_strength_range: tuple[float, float] = (0.3, 0.8),
    sparsity: float = 0.3,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Generate a correlation matrix with explicit control over positive vs negative correlations.

    This method creates a mathematically valid correlation matrix while allowing you to
    specify exactly what ratio of correlations should be positive vs negative.

    Args:
        num_features: Number of features
        positive_ratio: Fraction of non-zero correlations that should be positive (0.0 to 1.0)
        correlation_strength_range: (min, max) absolute strength of correlations
        sparsity: Fraction of correlations to set near zero (0.0 = dense, 1.0 = sparse)
        seed: Random seed for reproducibility

    Returns:
        Valid correlation matrix with controlled positive/negative ratio

    Example:
        # Create matrix where 30% of correlations are positive, 70% negative
        matrix = generate_mixed_correlation_matrix(
            num_features=10,
            positive_ratio=0.3,  # 30% positive, 70% negative
            correlation_strength_range=(0.4, 0.8),
            sparsity=0.2,
            seed=42
        )
    """

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # Start with identity matrix
    matrix = torch.eye(num_features)

    # Generate all possible pairs
    pairs = [(i, j) for i in range(num_features) for j in range(i + 1, num_features)]
    total_pairs = len(pairs)

    if total_pairs == 0:
        return matrix

    # Determine how many pairs to make sparse vs correlated
    num_sparse = int(total_pairs * sparsity)
    num_correlated = total_pairs - num_sparse

    if num_correlated == 0:
        return matrix

    # Select which pairs to correlate
    correlated_pairs = random.sample(pairs, num_correlated)

    # Determine positive vs negative
    num_positive = int(num_correlated * positive_ratio)

    # Assign correlations
    min_strength, max_strength = correlation_strength_range

    for i, (pair_i, pair_j) in enumerate(correlated_pairs):
        # Determine sign
        if i < num_positive:
            sign = 1
        else:
            sign = -1

        # Generate strength
        strength = random.uniform(min_strength, max_strength)
        correlation = sign * strength

        # Apply to matrix
        matrix[pair_i, pair_j] = correlation
        matrix[pair_j, pair_i] = correlation

    # Ensure the matrix is positive semi-definite using eigenvalue clipping
    eigenvals, eigenvecs = torch.linalg.eigh(matrix)
    eigenvals = torch.clamp(eigenvals, min=1e-6)  # Ensure positive
    matrix = eigenvecs @ torch.diag(eigenvals) @ eigenvecs.T

    # Renormalize to correlation matrix
    diag_sqrt = torch.sqrt(torch.diag(matrix))
    matrix = matrix / (diag_sqrt.unsqueeze(0) * diag_sqrt.unsqueeze(1))
    matrix.fill_diagonal_(1.0)

    return matrix
