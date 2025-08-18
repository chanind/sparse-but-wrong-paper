import torch

from sparse_but_wrong.toy_models.orthogonalize import orthogonalize


def test_orthogonalize_makes_vectors_orthogonal_if_possible():
    num_vectors = 10
    vector_len = 10
    target_cos_sim = 0
    embeddings = orthogonalize(num_vectors, vector_len, target_cos_sim)
    assert torch.allclose(embeddings @ embeddings.T, torch.eye(num_vectors), atol=0.05)


def test_orthogonalize_makes_vectors_have_target_cos_sim():
    num_vectors = 10
    vector_len = 10
    target_cos_sim = 0.2
    embeddings = orthogonalize(num_vectors, vector_len, target_cos_sim)
    # Calculate the cosine similarity matrix
    cos_sim_matrix = embeddings @ embeddings.T

    # Create a mask to ignore the diagonal (self-similarity)
    mask = torch.ones_like(cos_sim_matrix) - torch.eye(num_vectors)

    # Check that all off-diagonal elements are close to the target cosine similarity
    assert torch.allclose(
        cos_sim_matrix[mask.bool()],
        torch.full((num_vectors * (num_vectors - 1),), target_cos_sim),
        atol=0.05,
    )
