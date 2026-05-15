from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional

from .base import IROp
from ..state import TensorState, LocalSPMDType, ShardingSpec, Replicate


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
            dtype=x.dtype,
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

    @staticmethod
    def check_fp8_format_usage(
        tensor: TensorState,
        phase: str,
    ) -> Optional[str]:
        """Verify correct FP8 sub-format for the given phase.

        Convention: forward uses fp8e4m3 (higher precision),
                    backward uses fp8e5m2 (wider dynamic range).
        """
        if not tensor.is_fp8:
            return None

        if phase == "forward" and tensor.is_fp8e5m2:
            return (
                f"FP8 format violation: tensor '{tensor.name}' uses fp8e5m2 in "
                f"forward pass. Forward should use fp8e4m3 (higher precision). "
                f"fp8e5m2 is for backward (wider dynamic range for gradients)."
            )
        if phase == "backward" and tensor.is_fp8e4m3:
            return (
                f"FP8 format violation: tensor '{tensor.name}' uses fp8e4m3 in "
                f"backward pass. Backward should use fp8e5m2 (wider range). "
                f"fp8e4m3 is for forward (higher precision for activations)."
            )
        return None

    @staticmethod
    def check_fp8_scale_freshness(
        quantize_op_idx: int,
        amax_update_op_idx: Optional[int],
    ) -> Optional[str]:
        """Verify that scale was updated (via AmaxUpdate) before use.

        In delayed scaling, the scale for iteration N is derived from
        amax observed in iteration N-1. This check verifies that the
        AmaxUpdate from the previous iteration happens-before the
        FP8Quantize in the current iteration.
        """
        if amax_update_op_idx is None:
            return (
                f"FP8 scale freshness violation: FP8Quantize at op[{quantize_op_idx}] "
                f"uses a scale with no corresponding AmaxUpdate. "
                f"Delayed scaling requires amax observation before scale derivation."
            )

        if amax_update_op_idx >= quantize_op_idx:
            return (
                f"FP8 scale freshness violation: FP8Quantize at op[{quantize_op_idx}] "
                f"uses scale before AmaxUpdate at op[{amax_update_op_idx}]. "
                f"Scale must be derived from a PREVIOUS amax observation."
            )
        return None


@dataclass
class FP8Quantize(IROp):
    """Quantize a higher-precision tensor to FP8 with a symbolic scale.

    Forward:  dtype changes from src_dtype to dst_dtype (fp8e4m3 or fp8e5m2).
              Attaches scale_expr to the output TensorState.
    Backward: Dequantize (fp8 -> src_dtype) with the same scale.
    """
    x: str
    output: str
    scale_expr: str
    src_dtype: str
    dst_dtype: str

    def __post_init__(self):
        if self.dst_dtype not in ("fp8e4m3", "fp8e5m2"):
            raise ValueError(
                f"FP8Quantize dst_dtype must be 'fp8e4m3' or 'fp8e5m2', "
                f"got '{self.dst_dtype}'"
            )

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
                f"FP8Quantize: tensor '{x.name}' has dtype={x.dtype}, "
                f"expected src_dtype={self.src_dtype}"
            )
        result = replace(
            x,
            name=self.output,
            dtype=self.dst_dtype,
            fp8_scale_expr=self.scale_expr,
            expr=f"fp8_quantize({x.expr}, {self.scale_expr})" if x.expr else "",
        )
        ctx[self.output] = result
        return result

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        grad = replace(
            grad_output,
            name=f"grad_{self.x}",
            dtype=self.src_dtype,
            fp8_scale_expr=None,
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            expr=f"fp8_dequantize(grad({x.expr}), {self.scale_expr})" if x.expr else "",
        )
        return {self.x: grad}

    def clone_with_names(self, input_map, output_name):
        return FP8Quantize(
            x=input_map.get(self.x, self.x),
            output=output_name,
            scale_expr=self.scale_expr,
            src_dtype=self.src_dtype,
            dst_dtype=self.dst_dtype,
        )

    def __repr__(self):
        return (
            f"FP8Quantize({self.x}, {self.src_dtype}->{self.dst_dtype}, "
            f"scale={self.scale_expr}) -> {self.output}"
        )


@dataclass
class FP8Dequantize(IROp):
    """Dequantize an FP8 tensor back to higher precision.

    Forward:  dtype changes from src_dtype (fp8) to dst_dtype (fp32/bf16).
              Removes scale_expr from the output.
    Backward: Quantize gradient to fp8 with the same scale.
    """
    x: str
    output: str
    scale_expr: str
    src_dtype: str
    dst_dtype: str

    def __post_init__(self):
        if self.src_dtype not in ("fp8e4m3", "fp8e5m2"):
            raise ValueError(
                f"FP8Dequantize src_dtype must be 'fp8e4m3' or 'fp8e5m2', "
                f"got '{self.src_dtype}'"
            )

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
                f"FP8Dequantize: tensor '{x.name}' has dtype={x.dtype}, "
                f"expected src_dtype={self.src_dtype}"
            )
        result = replace(
            x,
            name=self.output,
            dtype=self.dst_dtype,
            fp8_scale_expr=None,
            expr=f"fp8_dequantize({x.expr}, {self.scale_expr})" if x.expr else "",
        )
        ctx[self.output] = result
        return result

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        grad = replace(
            grad_output,
            name=f"grad_{self.x}",
            dtype=self.src_dtype,
            fp8_scale_expr=self.scale_expr,
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            expr=f"fp8_quantize(grad({x.expr}), {self.scale_expr})" if x.expr else "",
        )
        return {self.x: grad}

    def clone_with_names(self, input_map, output_name):
        return FP8Dequantize(
            x=input_map.get(self.x, self.x),
            output=output_name,
            scale_expr=self.scale_expr,
            src_dtype=self.src_dtype,
            dst_dtype=self.dst_dtype,
        )

    def __repr__(self):
        return (
            f"FP8Dequantize({self.x}, {self.src_dtype}->{self.dst_dtype}, "
            f"scale={self.scale_expr}) -> {self.output}"
        )


@dataclass
class AmaxUpdate(IROp):
    """Record amax observation for delayed FP8 scaling.

    Models the delayed scaling pattern:
      1. Compute with current scale (derived from previous iteration's amax)
      2. Observe actual amax of the tensor AFTER computation
      3. Update amax_history buffer (used to derive scale for NEXT iteration)

    The output is a symbolic scalar representing the updated amax history.
    VJP returns empty (amax observation is not differentiable).
    """
    x: str
    output: str
    tensor_name: str
    iteration_expr: str = ""

    @property
    def input_names(self) -> List[str]:
        return [self.x]

    @property
    def output_name(self) -> str:
        return self.output

    def propagate_spmd_type(self, input_types):
        return LocalSPMDType.REPLICATE

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        x = ctx[self.x]
        if not (x.dtype and x.dtype.startswith("fp8")):
            raise ValueError(
                f"AmaxUpdate: expected FP8 input, got dtype={x.dtype} "
                f"for tensor '{x.name}'"
            )
        mesh = x.sharding.mesh
        rep_spec = ShardingSpec(
            placements=tuple(Replicate() for _ in mesh.shape),
            mesh=mesh,
        )
        result = TensorState(
            name=self.output,
            global_shape=(1,),
            local_shape=(1,),
            sharding=rep_spec,
            dtype="fp32",
            expr=f"amax({x.expr}, {self.iteration_expr})" if x.expr else f"amax({self.tensor_name})",
        )
        ctx[self.output] = result
        return result

    def vjp(self, ctx, grad_output):
        return {}

    def clone_with_names(self, input_map, output_name):
        return AmaxUpdate(
            x=input_map.get(self.x, self.x),
            output=output_name,
            tensor_name=self.tensor_name,
            iteration_expr=self.iteration_expr,
        )

    def __repr__(self):
        iter_str = f", iter={self.iteration_expr}" if self.iteration_expr else ""
        return f"AmaxUpdate({self.x}, tensor={self.tensor_name}{iter_str}) -> {self.output}"
