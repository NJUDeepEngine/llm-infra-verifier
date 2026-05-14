from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .base import IROp
from ..state import TensorState, LocalSPMDType


@dataclass
class Cast(IROp):
    """Cast tensor dtype without changing shape or sharding.

    Forward:  dtype changes from src_dtype to dst_dtype
    Backward: reverse cast (dst_dtype -> src_dtype)
    """
    x: str
    output: str
    src_dtype: str
    dst_dtype: str

    @property
    def input_names(self) -> List[str]:
        return [self.x]

    @property
    def output_name(self) -> str:
        return self.output

    def propagate_spmd_type(self, input_types):
        return input_types.get(self.x)

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        x = ctx[self.x]
        if x.dtype is not None and x.dtype != self.src_dtype:
            raise ValueError(
                f"Cast: tensor '{x.name}' has dtype={x.dtype}, "
                f"expected src_dtype={self.src_dtype}. "
                f"Op: Cast({self.x}, {self.src_dtype}->{self.dst_dtype}) -> {self.output}"
            )
        result = x.with_dtype(self.dst_dtype).with_name(self.output)
        result.expr = f"cast({x.expr}, {self.dst_dtype})" if x.expr else ""
        ctx[self.output] = result
        return result

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        grad = grad_output.with_dtype(self.src_dtype).with_name(f"grad_{self.x}")
        grad.expr = f"cast(grad({x.expr}), {self.src_dtype})" if x.expr else ""
        return {self.x: grad}

    def clone_with_names(self, input_map, output_name):
        return Cast(
            x=input_map.get(self.x, self.x),
            output=output_name,
            src_dtype=self.src_dtype,
            dst_dtype=self.dst_dtype,
        )

    def __repr__(self):
        return f"Cast({self.x}, {self.src_dtype}->{self.dst_dtype}) -> {self.output}"


@dataclass
class LossScale(IROp):
    """Scale or unscale loss for mixed-precision training.

    direction="scale":   multiply by scale factor (before backward)
    direction="unscale":  divide by scale factor (after backward, before optimizer step)

    Forward:  shape/sharding unchanged, expr annotated with scaling
    Backward: reverse direction (scale <-> unscale)
    """
    x: str
    output: str
    scale: float
    direction: str = "scale"

    def __post_init__(self):
        if self.direction not in ("scale", "unscale"):
            raise ValueError(f"LossScale direction must be 'scale' or 'unscale', got '{self.direction}'")

    @property
    def input_names(self) -> List[str]:
        return [self.x]

    @property
    def output_name(self) -> str:
        return self.output

    def propagate_spmd_type(self, input_types):
        return input_types.get(self.x)

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        x = ctx[self.x]
        op = "*" if self.direction == "scale" else "/"
        expr = f"({x.expr} {op} {self.scale})" if x.expr else ""

        from dataclasses import replace
        result = replace(x, name=self.output, expr=expr)
        ctx[self.output] = result
        return result

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        reverse_dir = "unscale" if self.direction == "scale" else "scale"
        op = "/" if self.direction == "scale" else "*"
        grad = TensorState(
            name=f"grad_{self.x}",
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            expr=f"({grad_output.expr} {op} {self.scale})" if grad_output.expr else "",
        )
        return {self.x: grad}

    def clone_with_names(self, input_map, output_name):
        return LossScale(
            x=input_map.get(self.x, self.x),
            output=output_name,
            scale=self.scale,
            direction=self.direction,
        )

    def __repr__(self):
        return f"LossScale({self.x}, {self.direction}, scale={self.scale}) -> {self.output}"


class DtypeGuard:
    """Static dtype validation rules for mixed-precision safety."""

    @staticmethod
    def check_allreduce_dtype(x: TensorState) -> Optional[str]:
        """FP16 AllReduce overflow risk. Returns warning message or None."""
        if x.is_fp16:
            return (
                f"AllReduce on FP16 tensor '{x.name}' risks overflow. "
                f"Consider casting to FP32 before AllReduce."
            )
        return None

    @staticmethod
    def check_matmul_dtype_match(a: TensorState, b: TensorState) -> Optional[str]:
        """MatMul operand dtype mismatch. Returns warning message or None."""
        if a.dtype != b.dtype and not (a.is_fp32 and b.is_fp32):
            return (
                f"MatMul dtype mismatch: '{a.name}' has dtype={a.dtype}, "
                f"'{b.name}' has dtype={b.dtype}. "
                f"Mixed-precision MatMul may produce unexpected results."
            )
        return None

    @staticmethod
    def check_collective_preserves_dtype(
        input_tensor: TensorState, output_tensor: TensorState,
    ) -> Optional[str]:
        """Collective ops should preserve dtype. Returns error message or None."""
        if input_tensor.dtype != output_tensor.dtype:
            return (
                f"Collective changed dtype: input '{input_tensor.name}' "
                f"dtype={input_tensor.dtype} -> output '{output_tensor.name}' "
                f"dtype={output_tensor.dtype}. Collectives must preserve dtype."
            )
        return None
