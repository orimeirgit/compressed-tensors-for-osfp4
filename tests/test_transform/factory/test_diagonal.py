# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.quantization import QuantizationConfig
from compressed_tensors.transform import (
    DiagonalFactory,
    TransformArgs,
    TransformConfig,
    TransformScheme,
    apply_transform_config,
)


def scheme(**kwargs):
    return TransformScheme(
        type="diagonal",
        precision=torch.bfloat16,
        apply=[TransformArgs(targets=["Linear"], location="input")],
        **kwargs,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_input_and_inverse_weight_preserve_biased_linear(dtype):
    torch.manual_seed(13)
    model = torch.nn.Sequential(torch.nn.Linear(16, 7, bias=True)).to(dtype)
    layer = model[0]
    x = torch.randn(2, 3, 16, dtype=dtype)
    expected = model(x)
    config = scheme()
    config.apply.append(
        TransformArgs(targets=["Linear"], location="weight_input", inverse=True)
    )
    factory = DiagonalFactory("smooth", config)
    scale = torch.tensor([0.5, 1, 2, 4] * 4, dtype=torch.bfloat16)
    factory.set_scale(layer, scale)
    factory.apply_to_model(model, use_tqdm=False)
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    assert layer.smooth_input.weight.shape == (16,)
    assert not hasattr(layer, "smooth_weight_input")


def test_equal_width_layers_have_independent_vectors():
    model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.Linear(16, 16))
    apply_transform_config(model, TransformConfig(config_groups={"smooth": scheme()}))
    model[0].smooth_input.set_scale(torch.full((16,), 2.0))
    assert torch.equal(model[1].smooth_input.weight, torch.ones(16))
    assert (
        model[0].smooth_input.weight.data_ptr()
        != model[1].smooth_input.weight.data_ptr()
    )


def test_round_trip_restores_vectors_without_rebalancing_weights():
    model = torch.nn.Sequential(torch.nn.Linear(16, 16))
    config = TransformConfig(config_groups={"smooth": scheme()})
    apply_transform_config(model, config)
    model[0].smooth_input.set_scale(torch.linspace(0.5, 2, 16))
    state = model.state_dict()
    restored = torch.nn.Sequential(torch.nn.Linear(16, 16))
    apply_transform_config(
        restored, TransformConfig.model_validate_json(config.model_dump_json())
    )
    restored.load_state_dict(state)
    x = torch.randn(2, 16)
    torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
    assert torch.equal(restored[0].weight, model[0].weight)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), 1e-44])
def test_invalid_and_unrepresentable_scales_rejected(value):
    layer = torch.nn.Linear(16, 4)
    with pytest.raises(ValueError, match="finite positive"):
        DiagonalFactory("smooth", scheme()).set_scale(layer, torch.full((16,), value))


@pytest.mark.parametrize("kind", ["missing", "shape", "dtype", "value"])
def test_corrupt_checkpoint_rejected_even_with_non_strict_loading(kind):
    model = torch.nn.Sequential(torch.nn.Linear(16, 4))
    apply_transform_config(model, TransformConfig(config_groups={"smooth": scheme()}))
    state = model.state_dict()
    key = "0.smooth_input.weight"
    if kind == "missing":
        del state[key]
    elif kind == "shape":
        state[key] = torch.ones(8, dtype=torch.bfloat16)
    elif kind == "dtype":
        state[key] = state[key].float()
    else:
        state[key] = torch.zeros(16, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        model.load_state_dict(state, strict=False)


@pytest.mark.parametrize(
    "kwargs", [{"head_dim": 8}, {"requires_grad": True}, {"randomize": True}]
)
def test_unsupported_diagonal_options_rejected(kwargs):
    with pytest.raises(ValueError):
        DiagonalFactory("smooth", scheme(**kwargs))


def test_loader_cannot_silently_discard_diagonal_metadata():
    with pytest.raises(ValueError, match="loader does not handle diagonal"):
        QuantizationConfig.model_validate(
            {
                "config_groups": {"NVFP4": ["Linear"]},
                "transform_config": TransformConfig(
                    config_groups={"smooth": scheme()}
                ).model_dump(),
            }
        )


def test_inverse_is_reciprocal_and_restores_input():
    layer = torch.nn.Linear(16, 4)
    factory = DiagonalFactory("smooth", scheme())
    factory.set_scale(layer, torch.linspace(0.5, 2, 16))
    transform = factory.create_transform(layer, scheme().apply[0])
    x = torch.randn(3, 16)
    torch.testing.assert_close(transform.right_inverse(transform(x)), x)
    assert not transform.args.inverse


def test_keyword_input_and_offloaded_scale():
    from compressed_tensors.offload import offload_module

    model = torch.nn.Sequential(torch.nn.Linear(16, 4))
    apply_transform_config(model, TransformConfig(config_groups={"smooth": scheme()}))
    transform = model[0].smooth_input
    offload_module(transform, onload_device="cpu", offload_device="cpu")
    transform.set_scale(torch.full((16,), 2.0))
    x = torch.randn(2, 16)
    expected = torch.nn.functional.linear(x * 2, model[0].weight, model[0].bias)
    torch.testing.assert_close(model[0](input=x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("targets", [["missing"], ["0", "Linear"]])
def test_missing_and_overlapping_targets_rejected_before_mutation(targets):
    model = torch.nn.Sequential(torch.nn.Linear(16, 4))
    config = scheme()
    config.apply[0].targets = targets
    with pytest.raises(ValueError):
        DiagonalFactory("smooth", config).apply_to_model(model)
    assert not hasattr(model[0], "smooth_input")


def test_repeated_application_rejected():
    model = torch.nn.Sequential(torch.nn.Linear(16, 4))
    config = TransformConfig(config_groups={"smooth": scheme()})
    apply_transform_config(model, config)
    with pytest.raises(ValueError, match="applied twice"):
        apply_transform_config(model, config)
