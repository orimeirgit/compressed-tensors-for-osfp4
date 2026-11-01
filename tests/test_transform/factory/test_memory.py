# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import Counter

import pytest
import torch
from compressed_tensors.offload import (
    disable_offloading,
    disable_onloading,
    set_onload_device,
)
from compressed_tensors.transform import (
    TransformArgs,
    TransformBase,
    TransformConfig,
    TransformScheme,
    apply_transform_config,
)
from tests.test_transform.conftest import TransformableModel
from tests.testing_utils import requires_gpu


@requires_gpu
@pytest.mark.parametrize("type", ("hadamard", "random-hadamard"))
@pytest.mark.parametrize("randomize", (True, False))
@pytest.mark.parametrize("requires_grad", (True, False))
# @pytest.mark.parametrize("offload", (True, False))
@pytest.mark.parametrize("offload", (True,))
def test_memory_sharing(type, randomize, requires_grad, offload):
    # load model (maybe with offloading)
    model = TransformableModel(2, 2, 4, 4, 8, 8)
    if offload:
        set_onload_device(model, torch.device("cuda"))

    # add transforms to model
    config = TransformConfig(
        config_groups={
            "": TransformScheme(
                type=type,
                randomize=randomize,
                requires_grad=requires_grad,
                apply=[
                    TransformArgs(targets="Linear", location="input"),
                    TransformArgs(targets="Linear", location="output"),
                ],
            )
        }
    )
    apply_transform_config(model, config)

    for context in disable_onloading, disable_offloading:
        with context():
            weights = [
                m.weight for m in model.modules() if isinstance(m, TransformBase)
            ]
            weight_to_count = Counter(weights)
            size_to_weight = {weight.size(0): weight for weight in weight_to_count}

            assert len(weight_to_count) == len(size_to_weight) == 3
            assert weight_to_count[size_to_weight[2]] == 3
            assert weight_to_count[size_to_weight[4]] == 4
            assert weight_to_count[size_to_weight[8]] == 3


def test_diagonal_equal_width_layers_have_independent_vectors():
    """Unlike Hadamard matrices, learned vectors must not share layer storage."""
    model = TransformableModel(16, 16, 16)
    config = TransformConfig(
        config_groups={
            "smooth_quant": TransformScheme(
                type="diagonal",
                precision=torch.bfloat16,
                apply=[TransformArgs(targets="Linear", location="input")],
            )
        }
    )
    apply_transform_config(model, config)
    first, second = (layer.smooth_quant_input for layer in model.fcs)
    first.set_scale(torch.full((16,), 2.0))
    assert first.weight.data_ptr() != second.weight.data_ptr()
    torch.testing.assert_close(
        second.weight, torch.ones(16, dtype=torch.bfloat16), rtol=0, atol=0
    )
