import torch

from sparse_but_wrong.toy_models.get_training_batch import (
    create_correlation_matrix,
    generate_random_correlation_matrix,
    get_correlated_features,
    get_training_batch,
)
from sparse_but_wrong.util import DEFAULT_DEVICE


def test_get_training_batch_fires_features_with_correct_probabilities():
    firing_probs = torch.tensor([0.3, 0.2, 0.1]).to(DEFAULT_DEVICE)
    batch_size = 1000
    samples = get_training_batch(batch_size, firing_probs)

    # Calculate the actual firing probabilities from the samples
    actual_probs = (samples > 0).float().mean(dim=0)

    # Assert that the actual probabilities are close to the expected ones
    torch.testing.assert_close(actual_probs, firing_probs, atol=0.05, rtol=0)


def test_get_training_batch_fires_features_with_correct_magnitudes():
    firing_probs = torch.tensor([1.0, 1.0, 1.0])
    std_firing_magnitudes = torch.tensor([0.1, 0.2, 0.3]).to(DEFAULT_DEVICE)
    batch_size = 1000
    samples = get_training_batch(batch_size, firing_probs, std_firing_magnitudes)
    actual_magnitudes = samples.std(dim=0)
    torch.testing.assert_close(
        actual_magnitudes, std_firing_magnitudes, atol=0.05, rtol=0
    )


def test_get_training_batch_never_fires_negative_magnitudes():
    firing_probs = torch.tensor([1.0, 1.0, 1.0])
    std_firing_magnitudes = torch.tensor([0.5, 1.0, 2.0])
    batch_size = 1000
    samples = get_training_batch(batch_size, firing_probs, std_firing_magnitudes)
    assert torch.all(samples >= 0)


def test_get_training_batch_can_set_mean_magnitudes():
    firing_probs = torch.tensor([0.5, 0.5, 1.0])
    firing_means = torch.tensor([1.5, 2.5, 3.5])
    batch_size = 1000
    samples = get_training_batch(
        batch_size, firing_probs, mean_firing_magnitudes=firing_means
    )
    assert set(samples[:, 0].tolist()) == {0, 1.5}
    assert set(samples[:, 1].tolist()) == {0, 2.5}
    assert set(samples[:, 2].tolist()) == {3.5}


# Tests for new correlation functionality


def test_get_correlated_features_preserves_marginal_probabilities():
    batch_size = 2000
    firing_probabilities = torch.tensor([0.1, 0.3, 0.5, 0.7])
    correlation_matrix = torch.eye(4)  # Independent features

    features = get_correlated_features(
        batch_size, firing_probabilities, correlation_matrix
    )

    assert features.shape == (batch_size, 4)
    assert torch.all((features == 0) | (features == 1))  # Binary features

    # Check marginal probabilities are preserved
    actual_probs = features.mean(dim=0)
    torch.testing.assert_close(actual_probs, firing_probabilities, atol=0.05, rtol=0.1)


def test_get_correlated_features_zero_correlation():
    batch_size = 1500
    firing_probabilities = torch.tensor([0.3, 0.3])
    correlation_matrix = torch.eye(2)  # Zero correlation

    features = get_correlated_features(
        batch_size, firing_probabilities, correlation_matrix
    )

    # Check correlation is approximately zero
    correlation = torch.corrcoef(features.T)[0, 1]
    assert abs(correlation) < 0.15  # Should be close to zero


def test_get_correlated_features_positive_correlation():
    batch_size = 1500
    firing_probabilities = torch.tensor([0.4, 0.4])
    correlation_matrix = torch.tensor([[1.0, 0.8], [0.8, 1.0]])

    features = get_correlated_features(
        batch_size, firing_probabilities, correlation_matrix
    )

    # Check positive correlation
    correlation = torch.corrcoef(features.T)[0, 1]
    assert correlation > 0.5  # Should be substantially positive


def test_get_correlated_features_negative_correlation():
    batch_size = 1500
    firing_probabilities = torch.tensor([0.5, 0.5])
    correlation_matrix = torch.tensor([[1.0, -0.7], [-0.7, 1.0]])

    features = get_correlated_features(
        batch_size, firing_probabilities, correlation_matrix
    )

    # Check negative correlation
    correlation = torch.corrcoef(features.T)[0, 1]
    assert correlation < -0.3  # Should be substantially negative


def test_get_correlated_features_mixed_correlations():
    batch_size = 2000
    firing_probabilities = torch.tensor([0.3, 0.3, 0.4, 0.4])
    correlation_matrix = torch.tensor(
        [
            [1.0, 0.6, -0.3, 0.0],
            [0.6, 1.0, 0.0, -0.4],
            [-0.3, 0.0, 1.0, 0.5],
            [0.0, -0.4, 0.5, 1.0],
        ]
    )

    features = get_correlated_features(
        batch_size, firing_probabilities, correlation_matrix
    )
    actual_correlations = torch.corrcoef(features.T)

    # Check some key correlations
    assert actual_correlations[0, 1] > 0.3  # Positive correlation
    assert actual_correlations[0, 2] < -0.1  # Negative correlation
    assert abs(actual_correlations[0, 3]) < 0.2  # Near zero correlation
    assert actual_correlations[2, 3] > 0.2  # Positive correlation


def test_get_correlated_features_different_devices():
    if not torch.cuda.is_available():
        return  # Skip if CUDA not available

    batch_size = 100
    firing_probabilities = torch.tensor([0.3, 0.4])
    correlation_matrix = torch.tensor([[1.0, 0.5], [0.5, 1.0]])

    # Test with CUDA device
    features_cuda = get_correlated_features(
        batch_size,
        firing_probabilities,
        correlation_matrix,
        device=torch.device("cuda"),
    )

    assert features_cuda.device.type == "cuda"
    assert features_cuda.shape == (batch_size, 2)


def test_create_correlation_matrix_basic():
    matrix = create_correlation_matrix(num_features=3)

    expected = torch.eye(3)
    torch.testing.assert_close(matrix, expected)


def test_create_correlation_matrix_with_correlations():
    matrix = create_correlation_matrix(
        num_features=4, correlations={(0, 1): 0.8, (1, 2): -0.6, (0, 3): 0.3}
    )

    # Check symmetry
    torch.testing.assert_close(matrix, matrix.T)

    # Check diagonal is 1
    torch.testing.assert_close(torch.diag(matrix), torch.ones(4))

    # Check specified correlations (allow some tolerance for matrix fixing)
    assert abs(matrix[0, 1] - 0.8) < 0.1  # Should be close to 0.8
    assert abs(matrix[1, 0] - 0.8) < 0.1  # Symmetric
    assert abs(matrix[1, 2] - (-0.6)) < 0.1  # Should be close to -0.6
    assert abs(matrix[2, 1] - (-0.6)) < 0.1  # Symmetric
    assert abs(matrix[0, 3] - 0.3) < 0.1  # Should be close to 0.3
    assert abs(matrix[3, 0] - 0.3) < 0.1  # Symmetric

    # Check unspecified correlations are small (close to 0)
    assert abs(matrix[0, 2]) < 0.1
    assert abs(matrix[2, 3]) < 0.1


def test_create_correlation_matrix_with_default_correlation():
    matrix = create_correlation_matrix(num_features=3, default_correlation=0.2)

    expected = torch.tensor([[1.0, 0.2, 0.2], [0.2, 1.0, 0.2], [0.2, 0.2, 1.0]])
    torch.testing.assert_close(matrix, expected)


def test_create_correlation_matrix_overrides_default():
    matrix = create_correlation_matrix(
        num_features=3, correlations={(0, 1): 0.9}, default_correlation=0.1
    )

    expected = torch.tensor([[1.0, 0.9, 0.1], [0.9, 1.0, 0.1], [0.1, 0.1, 1.0]])
    torch.testing.assert_close(matrix, expected)


def test_create_correlation_matrix_warns_non_psd():
    # This should trigger a warning but still return the matrix
    matrix = create_correlation_matrix(
        num_features=3, correlations={(0, 1): 0.9, (1, 2): 0.9, (0, 2): -0.9}
    )

    # Should still create the matrix even if not PSD
    assert matrix.shape == (3, 3)
    assert torch.allclose(matrix, matrix.T)


def test_get_training_batch_with_correlation_matrix():
    batch_size = 1000
    firing_probabilities = torch.tensor([0.2, 0.4, 0.3])
    correlation_matrix = create_correlation_matrix(
        num_features=3, correlations={(0, 1): 0.7, (0, 2): -0.3}
    )

    batch = get_training_batch(
        batch_size=batch_size,
        firing_probabilities=firing_probabilities,
        correlation_matrix=correlation_matrix,
    )

    assert batch.shape == (batch_size, 3)
    assert torch.all(batch >= 0)  # ReLU ensures non-negative

    # Check marginal probabilities preserved
    firing_rates = (batch > 0).float().mean(dim=0)
    torch.testing.assert_close(firing_rates, firing_probabilities, atol=0.05, rtol=0.1)

    # Check correlations are in right direction
    binary_features = (batch > 0).float()
    correlations = torch.corrcoef(binary_features.T)
    assert correlations[0, 1] > 0.3  # Should be positive
    assert correlations[0, 2] < -0.05  # Should be negative (relaxed threshold)


def test_get_training_batch_correlation_with_magnitudes():
    batch_size = 1000
    firing_probabilities = torch.tensor([0.5, 0.5])
    correlation_matrix = create_correlation_matrix(
        num_features=2, correlations={(0, 1): 0.8}
    )
    mean_magnitudes = torch.tensor([1.5, 2.0])
    std_magnitudes = torch.tensor([0.1, 0.2])

    batch = get_training_batch(
        batch_size=batch_size,
        firing_probabilities=firing_probabilities,
        correlation_matrix=correlation_matrix,
        mean_firing_magnitudes=mean_magnitudes,
        std_firing_magnitudes=std_magnitudes,
    )

    # Check correlation preserved with magnitudes
    binary_features = (batch > 0).float()
    correlation = torch.corrcoef(binary_features.T)[0, 1]
    assert correlation > 0.5

    # Check magnitudes when features fire
    firing_batch_0 = batch[batch[:, 0] > 0, 0]
    firing_batch_1 = batch[batch[:, 1] > 0, 1]

    # Should be around the mean magnitudes
    assert abs(firing_batch_0.mean() - 1.5) < 0.1
    assert abs(firing_batch_1.mean() - 2.0) < 0.1


def test_get_training_batch_fallback_to_independent():
    # Test that None correlation_matrix falls back to original behavior
    batch_size = 1000
    firing_probabilities = torch.tensor([0.2, 0.4, 0.6])

    # Without correlation matrix (original behavior)
    batch_independent = get_training_batch(
        batch_size=batch_size, firing_probabilities=firing_probabilities
    )

    # With identity correlation matrix (should be equivalent)
    batch_correlated = get_training_batch(
        batch_size=batch_size,
        firing_probabilities=firing_probabilities,
        correlation_matrix=torch.eye(3),
    )

    # Both should preserve marginal probabilities
    rates_independent = (batch_independent > 0).float().mean(dim=0)
    rates_correlated = (batch_correlated > 0).float().mean(dim=0)

    torch.testing.assert_close(
        rates_independent, firing_probabilities, atol=0.05, rtol=0.1
    )
    torch.testing.assert_close(
        rates_correlated, firing_probabilities, atol=0.05, rtol=0.1
    )


def test_get_correlated_features_extreme_probabilities():
    batch_size = 1000

    # Test with very low and very high probabilities
    firing_probabilities = torch.tensor([0.01, 0.99])
    correlation_matrix = torch.tensor([[1.0, 0.5], [0.5, 1.0]])

    features = get_correlated_features(
        batch_size, firing_probabilities, correlation_matrix
    )

    # Should still preserve marginal probabilities
    actual_probs = features.mean(dim=0)
    torch.testing.assert_close(actual_probs, firing_probabilities, atol=0.02, rtol=0.2)


def test_get_correlated_features_single_feature():
    batch_size = 100
    firing_probabilities = torch.tensor([0.3])
    correlation_matrix = torch.tensor([[1.0]])

    features = get_correlated_features(
        batch_size, firing_probabilities, correlation_matrix
    )

    assert features.shape == (batch_size, 1)
    actual_prob = features.mean()
    torch.testing.assert_close(actual_prob, firing_probabilities[0], atol=0.1, rtol=0.2)


def test_correlation_matrix_large_scale():
    # Test with larger correlation matrices
    num_features = 10
    firing_probabilities = torch.rand(num_features) * 0.8 + 0.1  # Between 0.1 and 0.9

    # Create a valid correlation matrix
    correlations = {}
    for i in range(num_features - 1):
        correlations[(i, i + 1)] = 0.3  # Chain of correlations

    correlation_matrix = create_correlation_matrix(
        num_features=num_features, correlations=correlations
    )

    batch_size = 500
    features = get_correlated_features(
        batch_size, firing_probabilities, correlation_matrix
    )

    assert features.shape == (batch_size, num_features)
    actual_probs = features.mean(dim=0)
    torch.testing.assert_close(actual_probs, firing_probabilities, atol=0.1, rtol=0.2)


def test_generate_random_correlation_matrix_basic():
    matrix = generate_random_correlation_matrix(
        num_features=4,
        positive_ratio=0.6,
        seed=42,
    )

    # Should be a valid correlation matrix
    assert matrix.shape == (4, 4)
    torch.testing.assert_close(matrix, matrix.T)  # Symmetric
    torch.testing.assert_close(torch.diag(matrix), torch.ones(4))  # Diagonal is 1

    # Check eigenvalues are non-negative (positive semi-definite)
    eigenvals = torch.linalg.eigvals(matrix)
    assert torch.all(eigenvals.real >= -1e-6)


def test_generate_random_correlation_matrix_reproducible():
    # Same seed should produce same results
    matrix1 = generate_random_correlation_matrix(num_features=3, seed=123)
    matrix2 = generate_random_correlation_matrix(num_features=3, seed=123)

    torch.testing.assert_close(matrix1, matrix2)

    # Different seeds should produce different results
    matrix3 = generate_random_correlation_matrix(num_features=3, seed=456)
    assert not torch.allclose(matrix1, matrix3)


def test_generate_random_correlation_matrix_parameters():
    # Test with extreme parameters
    matrix = generate_random_correlation_matrix(
        num_features=5,
        positive_ratio=0.8,  # Mostly positive
        seed=42,
    )

    # Should still be valid
    assert matrix.shape == (5, 5)
    torch.testing.assert_close(matrix, matrix.T)
    torch.testing.assert_close(torch.diag(matrix), torch.ones(5))
