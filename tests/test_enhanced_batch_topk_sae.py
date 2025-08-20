import torch
from sae_lens import BatchTopKTrainingSAE
from sae_lens.saes.sae import TrainStepInput

from sparse_but_wrong.enchanced_batch_topk_sae import (
    EnchancedBatchTopKTrainingSAE,
    EnhancedBatchTopKTrainingSAEConfig,
)
from tests.helpers import random_model_params


def test_EnchancedBatchTopkTrainingSAE_gives_same_results_after_folding_W_dec_norm_if_normalize_acts_by_decoder_norm():
    cfg = EnhancedBatchTopKTrainingSAEConfig(
        k=2,
        d_in=5,
        d_sae=10,
        normalize_acts_by_decoder_norm=True,
    )
    sae = EnchancedBatchTopKTrainingSAE(cfg)
    random_model_params(sae)

    test_input = torch.randn(300, 5)

    pre_fold_feats = sae.encode(test_input)
    pre_fold_output = sae.decode(pre_fold_feats)

    sae.fold_W_dec_norm()

    post_fold_feats = sae.encode(test_input)
    post_fold_output = sae.decode(post_fold_feats)

    assert torch.allclose(pre_fold_feats, post_fold_feats, rtol=1e-3)
    assert torch.allclose(pre_fold_output, post_fold_output, rtol=1e-3)


def test_EnchancedBatchTopkTrainingSAE_gives_same_results_as_BatchTopkTrainingSAE_if_decoder_is_already_normalized():
    cfg = EnhancedBatchTopKTrainingSAEConfig(
        k=2,
        d_in=5,
        d_sae=10,
        normalize_acts_by_decoder_norm=True,
    )
    sae = EnchancedBatchTopKTrainingSAE(cfg)
    random_model_params(sae)

    sae.fold_W_dec_norm()

    sae2 = BatchTopKTrainingSAE(cfg)
    with torch.no_grad():
        for param_name, param in sae2.named_parameters():
            param.data = getattr(sae, param_name).data

    test_input = torch.randn(300, 5)

    enhanced_acts = sae.encode(test_input)
    enhanced_output = sae.decode(enhanced_acts)

    batch_topk_acts = sae2.encode(test_input)
    batch_topk_output = sae2.decode(batch_topk_acts)

    assert torch.allclose(enhanced_acts, batch_topk_acts, rtol=1e-3)
    assert torch.allclose(enhanced_output, batch_topk_output, rtol=1e-3)


def test_EnchancedBatchTopkTrainingSAE_k_transitions_from_initial_to_final_over_duration():
    cfg = EnhancedBatchTopKTrainingSAEConfig(
        k=10,
        d_in=5,
        d_sae=20,
        initial_k=2,
        transition_k_duration_steps=8,
        transition_k_start_step=0,
    )
    sae = EnchancedBatchTopKTrainingSAE(cfg)
    random_model_params(sae)

    test_input = torch.randn(32, 5)
    train_input = TrainStepInput(
        sae_in=test_input, coefficients={}, dead_neuron_mask=None
    )

    # Check initial k value
    assert sae.activation_fn.k == 2  # type: ignore[attr-defined]

    # Step through training and check k values at key points
    expected_k_values = [
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        10,
    ]  # Linear interpolation from 2 to 10 over 8 steps

    for step, expected_k in enumerate(expected_k_values):
        if step > 0:  # First check is before any training steps
            sae.training_forward_pass(train_input)

        actual_k = sae.activation_fn.k  # type: ignore[attr-defined]
        assert actual_k == expected_k, (
            f"Step {step}: expected k={expected_k}, got k={actual_k}"
        )


def test_EnchancedBatchTopkTrainingSAE_k_stays_at_initial_before_start_step():
    cfg = EnhancedBatchTopKTrainingSAEConfig(
        k=10,
        d_in=5,
        d_sae=20,
        initial_k=3,
        transition_k_duration_steps=5,
        transition_k_start_step=3,  # Start transition at step 3
    )
    sae = EnchancedBatchTopKTrainingSAE(cfg)
    random_model_params(sae)

    test_input = torch.randn(32, 5)
    train_input = TrainStepInput(
        sae_in=test_input, coefficients={}, dead_neuron_mask=None
    )

    # Check that k stays at initial value before start step
    for step in range(4):  # Steps 0, 1, 2, 3
        actual_k = sae.activation_fn.k  # type: ignore[attr-defined]
        assert actual_k == 3, f"Step {step}: expected k=3 (initial), got k={actual_k}"

        if step < 3:  # Don't do training step after the last check
            sae.training_forward_pass(train_input)


def test_EnchancedBatchTopkTrainingSAE_k_stays_at_final_after_transition_completes():
    cfg = EnhancedBatchTopKTrainingSAEConfig(
        k=8,
        d_in=5,
        d_sae=20,
        initial_k=2,
        transition_k_duration_steps=4,
        transition_k_start_step=0,
    )
    sae = EnchancedBatchTopKTrainingSAE(cfg)
    random_model_params(sae)

    test_input = torch.randn(32, 5)
    train_input = TrainStepInput(
        sae_in=test_input, coefficients={}, dead_neuron_mask=None
    )

    # Run enough steps to complete the transition and go beyond
    for _ in range(
        10
    ):  # Transition completes at step 4, so steps 4-9 should all have k=8
        sae.training_forward_pass(train_input)

    # Check that k stays at final value after transition completes
    for _ in range(5):  # Check a few more steps
        actual_k = sae.activation_fn.k  # type: ignore[attr-defined]
        assert actual_k == 8, f"Expected k=8 (final), got k={actual_k}"
        sae.training_forward_pass(train_input)


def test_EnchancedBatchTopkTrainingSAE_k_transition_with_non_zero_start_step():
    cfg = EnhancedBatchTopKTrainingSAEConfig(
        k=12,
        d_in=5,
        d_sae=20,
        initial_k=4,
        transition_k_duration_steps=4,
        transition_k_start_step=2,  # Start transition at step 2
    )
    sae = EnchancedBatchTopKTrainingSAE(cfg)
    random_model_params(sae)

    test_input = torch.randn(32, 5)
    train_input = TrainStepInput(
        sae_in=test_input, coefficients={}, dead_neuron_mask=None
    )

    # Expected k values: steps 0-1 should be 4, then transition 4->12 over steps 2-5, then 12 onwards
    expected_k_by_step = {
        0: 4,
        1: 4,  # Before start step
        2: 4,
        3: 6,
        4: 8,
        5: 10,
        6: 12,  # During transition (step 2 starts with initial, step 6 reaches final)
        7: 12,
        8: 12,
        9: 12,  # After transition
    }

    for step in range(10):
        actual_k = sae.activation_fn.k  # type: ignore[attr-defined]
        expected_k = expected_k_by_step[step]
        assert actual_k == expected_k, (
            f"Step {step}: expected k={expected_k}, got k={actual_k}"
        )
        sae.training_forward_pass(train_input)


def test_EnchancedBatchTopkTrainingSAE_activation_function_k_updates_correctly():
    cfg = EnhancedBatchTopKTrainingSAEConfig(
        k=6,
        d_in=5,
        d_sae=20,
        initial_k=2,
        transition_k_duration_steps=2,
        transition_k_start_step=0,
    )
    sae = EnchancedBatchTopKTrainingSAE(cfg)
    random_model_params(sae)

    batch_size = 32
    test_input = torch.randn(batch_size, 5)
    train_input = TrainStepInput(
        sae_in=test_input, coefficients={}, dead_neuron_mask=None
    )

    # Test that the activation function's k value affects encoding behavior
    # At step 0, k should be 2
    assert sae.activation_fn.k == 2  # type: ignore[attr-defined]

    # Encode and check that total active features equals k * batch_size (BatchTopK selects top k*batch_size features)
    features_step0 = sae.encode(test_input)
    total_active_step0 = (features_step0 > 0).sum()
    assert total_active_step0 == 2 * batch_size, (
        f"Expected {2 * batch_size} total active features, got {total_active_step0}"
    )

    # Take a training step to move to k=4
    sae.training_forward_pass(train_input)
    assert sae.activation_fn.k == 4  # type: ignore[attr-defined]

    # Encode again and check that total active features equals 4 * batch_size
    features_step1 = sae.encode(test_input)
    total_active_step1 = (features_step1 > 0).sum()
    assert total_active_step1 == 4 * batch_size, (
        f"Expected {4 * batch_size} total active features, got {total_active_step1}"
    )

    # Take another training step to reach final k=6
    sae.training_forward_pass(train_input)
    assert sae.activation_fn.k == 6  # type: ignore[attr-defined]

    # Encode again and check that total active features equals 6 * batch_size
    features_step2 = sae.encode(test_input)
    total_active_step2 = (features_step2 > 0).sum()
    assert total_active_step2 == 6 * batch_size, (
        f"Expected {6 * batch_size} total active features, got {total_active_step2}"
    )


def test_EnchancedBatchTopkTrainingSAE_k_does_not_change_when_tweening_disabled():
    # Test with initial_k=None (tweening disabled)
    cfg = EnhancedBatchTopKTrainingSAEConfig(
        k=5,
        d_in=5,
        d_sae=20,
        initial_k=None,  # No tweening
        transition_k_duration_steps=10,
    )
    sae = EnchancedBatchTopKTrainingSAE(cfg)
    random_model_params(sae)

    test_input = torch.randn(32, 5)
    train_input = TrainStepInput(
        sae_in=test_input, coefficients={}, dead_neuron_mask=None
    )

    # k should stay at the configured value throughout training
    for step in range(10):
        actual_k = sae.activation_fn.k  # type: ignore[attr-defined]
        assert actual_k == 5, (
            f"Step {step}: expected k=5 (no tweening), got k={actual_k}"
        )
        sae.training_forward_pass(train_input)
