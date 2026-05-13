from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .base import IROp


@dataclass
class Program:
    """Container for a sequence of IR operations (forward or backward)."""
    name: str = ""
    ops: List[IROp] = field(default_factory=list)

    def add(self, op: IROp) -> Program:
        self.ops.append(op)
        return self

    @property
    def collectives(self) -> List[IROp]:
        return [op for op in self.ops if op.is_collective()]

    @property
    def p2p_ops(self) -> List[IROp]:
        return [op for op in self.ops if op.is_p2p()]

    @property
    def compute_ops(self) -> List[IROp]:
        return [op for op in self.ops if not op.is_collective()]

    def __iter__(self):
        return iter(self.ops)

    def __len__(self):
        return len(self.ops)

    def __getitem__(self, idx):
        return self.ops[idx]

    def validate_names(self) -> List[str]:
        """Check for duplicate output names in the op sequence."""
        errors: List[str] = []
        seen: Dict[str, int] = {}
        for i, op in enumerate(self.ops):
            oname = op.output_name
            if oname in seen:
                errors.append(
                    f"Duplicate output name '{oname}': "
                    f"op[{i}] ({type(op).__name__}) conflicts with "
                    f"op[{seen[oname]}] ({type(self.ops[seen[oname]]).__name__})"
                )
            seen[oname] = i
        return errors

    def __repr__(self):
        ops_str = "\n  ".join(repr(op) for op in self.ops)
        return f"Program({self.name}, {len(self.ops)} ops):\n  {ops_str}"


def ir_to_str(program: Program) -> str:
    """Pretty-print a program."""
    lines = [f"Program: {program.name}"]
    for i, op in enumerate(program.ops):
        lines.append(f"  [{i}] {op}")
    return "\n".join(lines)
