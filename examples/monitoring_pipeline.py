"""Minimal educational example.

Shows how to:
1. Define a tiny context that records operation names.
2. Register primitive + scan handlers.
3. Interpret a function (with jit + scan) while collecting metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from functools import partial as _partial  # alias to avoid name clash

import jax
import jax.numpy as jnp

from slub.handlers import ScanSemantics, default_primitive_handler, default_scan_handler
from slub.interpreter import Dispatcher, Interpreter, InterpreterContext, reinterpret


@_partial(jax.tree_util.register_dataclass, meta_fields=("tags",), data_fields=())
@dataclass(frozen=True)
class OpCtx(InterpreterContext):
    tags: tuple[str, ...] = ()

    # Minimal protocol methods
    def add_meta(self, key: str) -> "OpCtx":
        return OpCtx(self.tags + (key,))

    def add_value(self, v):
        return self  # unused here

    def push(self) -> "OpCtx":
        return self

    def pop(self) -> "OpCtx":
        return self


def tag(op: str):
    def h(interpreter, ctx: OpCtx, eqn, invals):
        ctx = ctx.add_meta(op)
        return default_primitive_handler(interpreter, ctx, eqn, invals)

    return h


def make_dispatcher():
    # Also handle scan by inserting a context tag then delegating
    def scan_h(interpreter, ctx, eqn, invals):
        ctx = ctx.add_meta("scan")
        return default_scan_handler(
            interpreter, ctx, eqn, invals, threading=ScanSemantics.RESULT
        )

    return Dispatcher(
        {
            jax.lax.sin_p: tag("sin"),
            jax.lax.add_p: tag("add"),
            "scan": scan_h,
        }
    )


def build(interpreter: Interpreter[OpCtx]):
    @partial(reinterpret, interpreter=interpreter)
    def fn(x):
        def body(carry, t):
            v = jnp.sin(t)
            return carry + v, v

        total, xs = jax.lax.scan(body, 0.0, x)
        return jnp.sin(total) + xs.sum()

    return jax.jit(fn)  # jit wraps reinterpret ok


def main():
    dispatcher = make_dispatcher()
    interpreter = Interpreter(dispatcher=dispatcher)
    run = build(interpreter)
    data = jnp.linspace(0.0, 1.0, 5)
    out, ctx = run(OpCtx(), data)
    print("Result:", out)
    print("Tags:", ctx.tags)


if __name__ == "__main__":
    main()
