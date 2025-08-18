import torch
from tqdm import tqdm


def orthogonalize(
    num_vectors: int,
    vector_len: int,
    target_cos_sim: float = 0,
    num_steps: int = 1000,
    lr: float = 0.01,
) -> torch.Tensor:
    "Try to make the embeddings have cos sime as close to target as possible"
    embeddings = torch.randn(num_vectors, vector_len)
    embeddings /= embeddings.norm(p=2, dim=1, keepdim=True)  # Normalize
    embeddings.requires_grad_(True)
    num_vectors = embeddings.shape[0]

    # Set up an Optimization loop to create nearly-perpendicular vectors
    optimizer = torch.optim.Adam([embeddings], lr=lr)  # type: ignore

    losses = []

    pbar = tqdm(range(num_steps))
    for step_num in pbar:
        optimizer.zero_grad()

        dot_products = embeddings @ embeddings.T
        # Punish deviation from orthogonal
        diff = dot_products - target_cos_sim
        diff.fill_diagonal_(0)
        loss = diff.pow(2).sum()
        # Extra incentive to keep rows normalized
        loss += num_vectors * (dot_products.diag() - 1).pow(2).sum()

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pbar.set_description(f"loss: {loss.item():.3f}")
    embeddings = (
        (embeddings / embeddings.norm(p=2, dim=1, keepdim=True)).detach().clone()
    )
    embeddings.requires_grad_(False)
    return embeddings.detach().clone()
