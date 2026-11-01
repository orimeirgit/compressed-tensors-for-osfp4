# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch
from compressed_tensors.offload import set_onload_device
from compressed_tensors.transform import (
    TransformArgs,
    TransformConfig,
    TransformScheme,
    apply_transform_config,
)
from safetensors import safe_open
from tests.test_transform.conftest import TransformableModel
from tests.testing_utils import requires_gpu
from transformers import AutoModelForCausalLM, AutoTokenizer


@pytest.mark.parametrize("type", ("hadamard", "random-hadamard"))
@pytest.mark.parametrize("randomize", (True, False))
@pytest.mark.parametrize("offload", (True, False))
def test_serialization(type, randomize, model_apply, tmp_path, offload):
    # get model, maybe offload
    model, apply = model_apply
    if offload:
        set_onload_device(model, torch.device("cuda"))

    # apply transforms to model
    config = TransformConfig(
        config_groups={"": TransformScheme(type=type, randomize=randomize, apply=apply)}
    )
    apply_transform_config(model, config)

    # save model
    model_path = os.path.join(tmp_path, "test_model_path")
    model.save_pretrained(model_path)

    # check that saved values match model values
    # note that shared weights are only serialized once
    safetensors_path = os.path.join(model_path, "model.safetensors")
    device = "cuda:0" if offload else "cpu"
    with safe_open(safetensors_path, framework="pt", device=device) as file:
        saved_keys = set(file.keys())
        assert {
            "fcs.0.weight",
            "fcs.1.weight",
            "fcs.2.weight",
            "fcs.3.weight",
            "fcs.4.weight",
        } <= saved_keys
        for key in saved_keys:
            param = model.get_parameter(key)
            saved_param = file.get_tensor(key)
            assert torch.equal(param, saved_param)


@pytest.mark.skip("Requires transformers#40673")
@requires_gpu
@pytest.mark.parametrize(
    "model_stub,exp_perplexity",
    [
        ("nm-testing/Llama-3.2-1B-Instruct-spinquantR1R2R4-w4a16", 10.0),
        ("nm-testing/Llama-3.2-1B-Instruct-quip-w4a16", 10.0),
    ],
)
def test_load_perplexity(model_stub, exp_perplexity):
    model = AutoModelForCausalLM.from_pretrained(model_stub, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_stub)

    prompt = "The capital of France is Paris, the capital of Germany is Berlin"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    labels = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)

    perplexity = torch.exp(outputs.loss)
    assert perplexity <= exp_perplexity


@pytest.mark.parametrize("inverse", (False, True))
def test_serialization_diagonal(tmp_path, inverse):
    """Save and reload independent scales without changing balanced weights."""
    torch.manual_seed(13)
    model = TransformableModel(16, 16, 16).to(torch.bfloat16)
    config = TransformConfig(
        config_groups={
            "smooth_quant": TransformScheme(
                type="diagonal",
                precision=torch.bfloat16,
                apply=[
                    TransformArgs(targets="Linear", location="input", inverse=inverse)
                ],
            )
        }
    )
    apply_transform_config(model, config)
    for index, layer in enumerate(model.fcs):
        layer.smooth_quant_input.set_scale(torch.linspace(0.5, 2, 16) + index)
    model.save_pretrained(tmp_path)
    config_path = tmp_path / "transform_config.json"
    config_path.write_text(config.model_dump_json())

    with safe_open(tmp_path / "model.safetensors", framework="pt") as file:
        state = {key: file.get_tensor(key) for key in file.keys()}
    assert set(state) == {
        f"fcs.{index}.{parameter}"
        for index in range(2)
        for parameter in ("weight", "smooth_quant_input.weight")
    }
    for key, value in state.items():
        torch.testing.assert_close(value, model.get_parameter(key), rtol=0, atol=0)

    restored = TransformableModel(16, 16, 16).to(torch.bfloat16)
    apply_transform_config(
        restored, TransformConfig.model_validate_json(config_path.read_text())
    )
    restored.load_state_dict(state, strict=True)
    inputs = torch.randn(2, 16, dtype=torch.bfloat16)
    torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
