# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.transform import (
    TransformArgs,
    TransformLocation,
    TransformScheme,
)
from compressed_tensors.transform.factory.base import TransformBase, TransformFactory
from compressed_tensors.utils import (
    get_offloaded_device,
    match_named_modules,
    register_offload_module,
    update_offload_parameter,
)
from torch import Tensor
from torch.nn import Linear, Module, Parameter


__all__ = ["DiagonalFactory", "DiagonalTransform"]


@TransformFactory.register("diagonal")
class DiagonalFactory(TransformFactory):
    """Per-Linear input-channel transforms with independent, positive vectors.

    Input and weight-input transforms of the same layer use the same vector;
    equal-width layers do not share parameters. Set a learned vector with
    ``set_scale`` before applying the factory, or update the installed transform
    with ``set_scale`` during calibration. Identity is the initialization.
    """

    def __init__(self, name: str, scheme: TransformScheme, seed: int | None = None):
        super().__init__(name, scheme, seed)
        if scheme.head_dim is not None or scheme.randomize or scheme.requires_grad:
            raise ValueError(
                "Diagonal transforms require full input-channel vectors, "
                "randomize=False and requires_grad=False"
            )
        if scheme.precision not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            raise ValueError("Diagonal transform precision must be floating point")
        self.weights: dict[Module, Parameter] = {}

    def apply_to_model(self, model: Module, use_tqdm=True):
        # Check every target before mutating weights or installing hooks. A stale
        # checkpoint target must not silently omit its required input scaling.
        seen = set()
        for args in self.scheme.apply:
            if args.location not in (
                TransformLocation.INPUT,
                TransformLocation.WEIGHT_INPUT,
            ):
                raise ValueError(
                    "Diagonal transforms support input and weight_input only"
                )
            for target in args.targets:
                matches = list(match_named_modules(model, [target], args.ignore))
                if not matches:
                    raise ValueError(
                        f"Diagonal transform target {target!r} matched no layers"
                    )
                for _, module in matches:
                    key = (module, args.location)
                    if key in seen or hasattr(module, f"{self.name}_{args.location}"):
                        raise ValueError("A diagonal transform cannot be applied twice")
                    if not isinstance(module, Linear):
                        raise ValueError(
                            "Diagonal transforms support Linear layers only"
                        )
                    seen.add(key)
        super().apply_to_model(model, use_tqdm)

    def _apply_to_module(self, model: Module, module: Module, args: TransformArgs):
        if args.location != TransformLocation.INPUT:
            return super()._apply_to_module(model, module, args)
        transform = self.create_transform(module, args)
        register_offload_module(module, f"{self.name}_{args.location}", transform)

        def input_hook(_, inputs, kwargs):
            if inputs:
                return (transform(inputs[0]), *inputs[1:]), kwargs
            return inputs, {**kwargs, "input": transform(kwargs["input"])}

        module.register_forward_pre_hook(input_hook, prepend=True, with_kwargs=True)

    def set_scale(self, module: Linear, scale: Tensor) -> None:
        """Set the vector before attaching transforms to a layer."""
        if module in self.weights:
            raise ValueError("Set the scale on the installed DiagonalTransform instead")
        if not isinstance(module, Linear):
            raise ValueError("Diagonal transforms currently support Linear layers only")
        scale = scale.detach().to(
            device=get_offloaded_device(module), dtype=self.scheme.precision
        )
        DiagonalTransform.validate_scale(scale, module.in_features)
        self.weights[module] = Parameter(scale.clone(), requires_grad=False)

    def create_transform(self, module: Module, args: TransformArgs):
        if not isinstance(module, Linear) or args.location not in (
            TransformLocation.INPUT,
            TransformLocation.WEIGHT_INPUT,
        ):
            raise ValueError(
                "Diagonal transforms support Linear input and weight_input only"
            )
        if module not in self.weights:
            self.weights[module] = Parameter(
                torch.ones(
                    module.in_features,
                    device=get_offloaded_device(module),
                    dtype=self.scheme.precision,
                ),
                requires_grad=False,
            )
        return DiagonalTransform(
            self.weights[module], self.scheme, args, module.in_features
        )


class DiagonalTransform(TransformBase):
    """Apply a vector or its reciprocal along the last tensor dimension.

    No dense matrix, normalization, or transpose is involved. The stored vector
    is always the forward multiplier, including for inverse transforms.
    """

    def __init__(
        self, weight: Parameter, scheme: TransformScheme, args: TransformArgs, size: int
    ):
        super().__init__()
        self.weight = weight
        self.scheme = scheme
        self.args = args
        self.size = size
        self._validated = False

    @staticmethod
    def validate_scale(scale: Tensor, size: int) -> None:
        if scale.shape != (size,) or not scale.is_floating_point():
            raise ValueError(
                f"Diagonal scale must be a floating vector of shape {(size,)}"
            )
        if not scale.is_meta and (
            not torch.isfinite(scale).all() or not (scale > 0).all()
        ):
            raise ValueError("Diagonal scale must contain finite positive values")

    def set_scale(self, scale: Tensor) -> None:
        """Install a learned vector, validating after rounding to stored precision."""
        scale = scale.detach().to(dtype=self.scheme.precision)
        self.validate_scale(scale, self.size)
        update_offload_parameter(self, "weight", scale)
        self._validated = False

    def forward(self, value: Tensor) -> Tensor:
        if not self._validated:
            self.validate_scale(self.weight, self.size)
            self._validated = True
        if value.shape[-1] != self.size:
            raise ValueError(f"Diagonal transform expected input width {self.size}")
        scale = self.weight.to(value)
        return value / scale if self.args.inverse else value * scale

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        key = prefix + "weight"
        if key not in state_dict:
            error_msgs.append(f"Required diagonal transform scale {key!r} is missing")
        else:
            try:
                self.validate_scale(state_dict[key], self.size)
                if state_dict[key].dtype != self.scheme.precision:
                    raise ValueError(
                        f"Diagonal scale dtype must be {self.scheme.precision}"
                    )
            except ValueError as error:
                error_msgs.append(f"{key}: {error}")
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._validated = False
