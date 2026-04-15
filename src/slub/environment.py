"""
Copyright 2025 CuspAI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from collections import defaultdict
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
from jax import Array
from jax.core import Atom
from jax.extend.core import Jaxpr, JaxprEqn, Literal, Var


@runtime_checkable
class _HasDTypeAndShape(Protocol):
    @property
    def dtype(self) -> Any: ...

    @property
    def shape(self) -> tuple[int, ...]: ...


class InterpreterEnvironment:
    env: dict[Var, Any]
    reference_counter: dict[Var, int]
    use_reference_counting: bool

    def __init__(
        self,
        jaxpr: Jaxpr,
        consts: Sequence[Array],
        args: Any,
        *,
        use_reference_counting: bool = True,
    ):
        self.env = {}
        self.reference_counter = defaultdict(int)
        self.use_reference_counting = use_reference_counting
        for v in jaxpr.invars + jaxpr.constvars:
            if isinstance(v, Literal):
                continue
            self.reference_counter[v] += 1
        eqn: JaxprEqn
        for eqn in jaxpr.eqns:
            for v in eqn.invars:
                if isinstance(v, Literal):
                    continue
                self.reference_counter[v] += 1
        for v in jaxpr.outvars:
            if isinstance(v, Literal):
                continue
            self.reference_counter[v] = np.iinfo(np.int32).max
        self.write_many(jaxpr.constvars, consts)
        self.write_many(jaxpr.invars, args)

    def __repr__(self) -> str:
        return f"InterpreterEnvironment({self.env})"

    def read(self, var: Atom) -> Any:
        if isinstance(var, Literal):
            return var.val
        self.reference_counter[var] -= 1
        result = self.env[var]
        if self.use_reference_counting and self.reference_counter[var] == 0:
            del self.env[var]
            del self.reference_counter[var]
        return result

    def write(self, var: Var, val: Any):
        if isinstance(val, _HasDTypeAndShape) and isinstance(
            var.aval, _HasDTypeAndShape
        ):
            if var.aval.dtype != val.dtype:
                raise ValueError(
                    f"Type mismatch when writing to env: var {var} has dtype {var.aval.dtype}, but value has dtype {val.dtype}"
                )
            if var.aval.shape != val.shape:
                raise ValueError(
                    f"Shape mismatch when writing to env: var {var} has shape {var.aval.shape}, but value has shape {val.shape}"
                )
        if not self.use_reference_counting or self.reference_counter[var] > 0:
            self.env[var] = val

    def read_many(self, vars: Sequence[Atom]) -> list[Any]:
        return list(map(self.read, vars))

    def write_many(self, vars: Sequence[Var], vals: Sequence[Any]):
        if not len(vars) == len(vals):
            raise ValueError(
                f"Length of vars {len(vars)} does not match length of vals {len(vals)}"
            )
        return list(map(self.write, vars, vals))
