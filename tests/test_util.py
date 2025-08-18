import torch

from sparse_but_wrong.util import (
    Tween,
    cos_sims,
    scale_grad,
    scale_parallel_gradients,
)


def test_cos_sims_calculates_cosine_similarity_correctly() -> None:
    # Create two test matrices with known cosine similarities
    mat1 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).T  # 2x3 matrix
    mat2 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).T  # 2x3 matrix

    result = cos_sims(mat1, mat2)

    # Expected cosine similarities:
    # [1.0, 0.0, 0.707]
    # [0.0, 1.0, 0.707]
    # [0.707, 0.707, 1.0]
    expected = torch.tensor([[1.0, 0.0, 0.707], [0.0, 1.0, 0.707], [0.707, 0.707, 1.0]])

    # Check shape
    assert result.shape == (3, 3)

    # Check values match expected within tolerance
    torch.testing.assert_close(result, expected, atol=1e-3, rtol=0)


def test_Tween_returns_start_value_before_start_step() -> None:
    tween = Tween(start=0.0, end=1.0, n_steps=10, start_step=5)
    assert tween(step=0) == 0.0
    assert tween(step=4) == 0.0


def test_Tween_returns_end_value_after_end_step() -> None:
    tween = Tween(start=0.0, end=1.0, n_steps=10, start_step=5)
    assert tween(step=16) == 1.0
    assert tween(step=20) == 1.0


def test_Tween_interpolates_linearly_during_steps() -> None:
    tween = Tween(start=0.0, end=1.0, n_steps=10, start_step=0)

    # Test midpoint
    assert tween(step=5) == 0.5

    # Test quarter points
    assert tween(step=2) == 0.2
    assert tween(step=7) == 0.7


def test_Tween_handles_negative_to_positive_range() -> None:
    tween = Tween(start=-1.0, end=1.0, n_steps=4, start_step=0)
    assert tween(step=0) == -1.0
    assert tween(step=2) == 0.0
    assert tween(step=4) == 1.0


def test_Tween_handles_non_zero_start_step() -> None:
    tween = Tween(start=0.0, end=1.0, n_steps=10, start_step=5)
    assert tween(step=3) == 0.0
    assert tween(step=10) == 0.5  # Halfway through the tween
    assert tween(step=15) == 1.0  # End of tween


def test_Tween_handles_int_start_and_end() -> None:
    tween = Tween(start=0, end=1, n_steps=10, start_step=5)
    assert tween(step=3) == 0.0
    assert tween(step=10) == 0.5  # Halfway through the tween
    assert tween(step=15) == 1.0  # End of tween


def test_Tween_list_returns_start_values_before_start_step() -> None:
    tween = Tween(start=[0.0, 1.0], end=[2.0, 3.0], n_steps=10, start_step=5)
    assert tween(step=0) == [0.0, 1.0]
    assert tween(step=4) == [0.0, 1.0]


def test_Tween_list_returns_end_values_after_end_step() -> None:
    tween = Tween(start=[0.0, 1.0], end=[2.0, 3.0], n_steps=10, start_step=5)
    assert tween(step=16) == [2.0, 3.0]
    assert tween(step=20) == [2.0, 3.0]


def test_Tween_list_interpolates_linearly_during_steps() -> None:
    tween = Tween(start=[0.0, 2.0], end=[1.0, 4.0], n_steps=10, start_step=0)

    # Test midpoint
    assert tween(step=5) == [0.5, 3.0]

    # Test quarter points
    assert tween(step=2) == [0.2, 2.4]
    assert tween(step=7) == [0.7, 3.4]


def test_Tween_list_handles_negative_to_positive_range() -> None:
    tween = Tween(start=[-1.0, -2.0], end=[1.0, 2.0], n_steps=4, start_step=0)
    assert tween(step=0) == [-1.0, -2.0]
    assert tween(step=2) == [0.0, 0.0]
    assert tween(step=4) == [1.0, 2.0]


def test_Tween_list_handles_non_zero_start_step() -> None:
    tween = Tween(start=[0.0, 10.0], end=[1.0, 20.0], n_steps=10, start_step=5)
    assert tween(step=3) == [0.0, 10.0]
    assert tween(step=10) == [0.5, 15.0]  # Halfway through the tween
    assert tween(step=15) == [1.0, 20.0]  # End of tween


def test_Tween_list_handles_different_ranges_per_element() -> None:
    tween = Tween(
        start=[0.0, 100.0, -50.0], end=[10.0, 200.0, 50.0], n_steps=5, start_step=0
    )

    # At step 1 (20% through)
    result = tween(step=1)
    assert result == [2.0, 120.0, -30.0]

    # At step 3 (60% through)
    result = tween(step=3)
    assert result == [6.0, 160.0, 10.0]


def test_Tween_list_handles_single_element_list() -> None:
    tween = Tween(start=[5.0], end=[15.0], n_steps=10, start_step=0)
    assert tween(step=0) == [5.0]
    assert tween(step=5) == [10.0]
    assert tween(step=10) == [15.0]


def test_Tween_list_with_zero_change_elements() -> None:
    tween = Tween(start=[1.0, 5.0, 3.0], end=[1.0, 10.0, 3.0], n_steps=4, start_step=0)

    # Elements 0 and 2 should remain constant, element 1 should change
    assert tween(step=0) == [1.0, 5.0, 3.0]
    assert tween(step=2) == [1.0, 7.5, 3.0]
    assert tween(step=4) == [1.0, 10.0, 3.0]


def test_scale_grad_1_does_nothing() -> None:
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    z = scale_grad(x, 1.0) @ y
    z.backward()
    assert x.grad is not None
    assert y.grad is not None
    assert x.grad.tolist() == [1.0, 2.0, 3.0]
    assert y.grad.tolist() == [1.0, 2.0, 3.0]


def test_scale_grad_does_not_affect_other_tensors() -> None:
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    z = scale_grad(x, 0.5) @ y
    z.backward()
    assert x.grad is not None
    assert y.grad is not None
    assert x.grad.tolist() == [0.5, 1.0, 1.5]
    assert y.grad.tolist() == [1.0, 2.0, 3.0]


def test_remove_parallel_gradients_removes_parallel_component() -> None:
    # Create input tensor with specific values
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)

    # Create a loss that would produce gradients parallel to x along dim=1
    # Use sum to create gradients that are parallel to the input
    y = scale_parallel_gradients(x, dim=1, scale=0.0)
    loss = (y * x.detach()).sum()
    loss.backward()

    assert x.grad is not None
    # After removing parallel gradients, dot product along dim=1 should be zero
    dot_products = (x.grad * x).sum(dim=1)
    torch.testing.assert_close(
        dot_products, torch.zeros_like(dot_products), atol=1e-6, rtol=0
    )


def test_scale_parallel_gradients_preserves_perpendicular_components() -> None:
    # Create input and a vector perpendicular to it
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    # Perpendicular vector: rotate 90 degrees
    perp = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])

    y = scale_parallel_gradients(x, dim=1, scale=0.0)
    # Create loss using perpendicular vector to produce perpendicular gradients
    loss = (y * perp).sum()
    loss.backward()

    assert x.grad is not None
    # The perpendicular component should be preserved
    # x.grad should be approximately equal to perp since there's no parallel component to remove
    torch.testing.assert_close(x.grad, perp, atol=1e-6, rtol=0)


def test_scale_parallel_gradients_forward_pass_unchanged() -> None:
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = scale_parallel_gradients(x, dim=1, scale=0.0)

    # Forward pass should return input unchanged
    torch.testing.assert_close(y, x, atol=1e-10, rtol=0)


def test_scale_parallel_gradients_with_different_dimensions() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)

    # Test with dim=0
    y = scale_parallel_gradients(x, dim=0, scale=0.0)
    loss = (y * x.detach()).sum()
    loss.backward()

    assert x.grad is not None
    # Dot product along dim=0 should be zero
    dot_products = (x.grad * x).sum(dim=0)
    torch.testing.assert_close(
        dot_products, torch.zeros_like(dot_products), atol=1e-6, rtol=0
    )


def test_remove_parallel_gradients_with_3d_tensor() -> None:
    x = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]], requires_grad=True
    )

    y = scale_parallel_gradients(x, dim=2, scale=0.0)
    loss = (y * x.detach()).sum()
    loss.backward()

    assert x.grad is not None
    # Dot product along dim=2 should be zero
    dot_products = (x.grad * x).sum(dim=2)
    torch.testing.assert_close(
        dot_products, torch.zeros_like(dot_products), atol=1e-6, rtol=0
    )


def test_scale_parallel_gradients_mixed_parallel_perpendicular() -> None:
    # Create a 2D input
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)

    y = scale_parallel_gradients(x, dim=1, scale=0.0)
    # Create a loss that has both parallel and perpendicular components
    # Parallel: x itself, Perpendicular: [[0, 1], [-1, 0]]
    parallel_component = x.detach()
    perpendicular_component = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
    mixed_target = parallel_component + perpendicular_component

    loss = (y * mixed_target).sum()
    loss.backward()

    assert x.grad is not None
    # The parallel component should be removed, leaving only perpendicular
    torch.testing.assert_close(x.grad, perpendicular_component, atol=1e-6, rtol=0)

    # Verify dot product is zero (no parallel component)
    dot_products = (x.grad * x).sum(dim=1)
    torch.testing.assert_close(
        dot_products, torch.zeros_like(dot_products), atol=1e-6, rtol=0
    )


def test_scale_parallel_gradients_scale_1_keeps_unchanged() -> None:
    # Create input tensor
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)

    # Apply scale_parallel_gradients with scale=1.0 (should not change anything)
    y = scale_parallel_gradients(x, dim=1, scale=1.0)
    loss = (y * x.detach()).sum()
    loss.backward()

    assert x.grad is not None
    # With scale=1.0, gradients should be the same as if we didn't use scale_parallel_gradients
    expected_grad = x.detach()
    torch.testing.assert_close(x.grad, expected_grad, atol=1e-6, rtol=0)


def test_scale_parallel_gradients_scale_half_reduces_parallel_component() -> None:
    # Create input tensor
    x = torch.tensor([[2.0, 0.0], [0.0, 3.0]], requires_grad=True)

    # Apply scale_parallel_gradients with scale=0.5
    y = scale_parallel_gradients(x, dim=1, scale=0.5)
    # Create loss with only parallel component (using x itself)
    loss = (y * x.detach()).sum()
    loss.backward()

    assert x.grad is not None
    # With scale=0.5, the parallel component should be scaled by 0.5
    expected_grad = 0.5 * x.detach()
    torch.testing.assert_close(x.grad, expected_grad, atol=1e-6, rtol=0)


def test_scale_parallel_gradients_scale_preserves_perpendicular() -> None:
    # Create input and perpendicular vector
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    perpendicular_component = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])

    # Apply scale_parallel_gradients with scale=0.3
    y = scale_parallel_gradients(x, dim=1, scale=0.3)
    # Create loss with mixed parallel and perpendicular components
    parallel_component = x.detach()
    mixed_target = parallel_component + perpendicular_component

    loss = (y * mixed_target).sum()
    loss.backward()

    assert x.grad is not None
    # Expected: perpendicular unchanged + parallel scaled by 0.3
    expected_grad = perpendicular_component + 0.3 * parallel_component
    torch.testing.assert_close(x.grad, expected_grad, atol=1e-6, rtol=0)


def test_scale_parallel_gradients_scale_2_amplifies_parallel_component() -> None:
    # Create input tensor
    x = torch.tensor([[1.0, 0.0], [0.0, 2.0]], requires_grad=True)

    # Apply scale_parallel_gradients with scale=2.0 (amplify parallel)
    y = scale_parallel_gradients(x, dim=1, scale=2.0)
    # Create loss with only parallel component
    loss = (y * x.detach()).sum()
    loss.backward()

    assert x.grad is not None
    # With scale=2.0, the parallel component should be doubled
    expected_grad = 2.0 * x.detach()
    torch.testing.assert_close(x.grad, expected_grad, atol=1e-6, rtol=0)
