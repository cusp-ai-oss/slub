# Slub

Minimal JAX interpreter layer for threading a custom context through computations.

## Installation

```bash
git clone https://github.com/cusp-ai-oss/slub.git
cd slub
uv sync
```

## Quick Start

```python
import jax, jax.numpy as jnp
from dataclasses import dataclass
from functools import partial as _partial
from functools import partial
from slub.interpreter import Interpreter, Dispatcher, InterpreterContext, reinterpret
from slub.handlers import default_primitive_handler

@_partial(jax.tree_util.register_dataclass, meta_fields=("tags",), data_fields=())
@dataclass(frozen=True)
class OpCtx(InterpreterContext):
    # 'tags' marked meta so it is NOT traced or differentiated over
    tags: tuple[str, ...] = ()
    def add_meta(self, k: str): return OpCtx(self.tags + (k,))
    def add_value(self, v): return self  # no value collection here
    def push(self): return self
    def pop(self): return self

def tag(name: str):
    def h(interpreter, ctx, eqn, invals):
        ctx = ctx.add_meta(name)
        return default_primitive_handler(interpreter, ctx, eqn, invals)
    return h

dispatcher = Dispatcher({jax.lax.sin_p: tag("sin"), jax.lax.add_p: tag("add")})
interp = Interpreter(dispatcher=dispatcher)

@partial(reinterpret, interpreter=interp)
def f(xs):
    return jnp.sin(xs).sum()

out, ctx = f(OpCtx(), jnp.linspace(0., 1., 5))
print(out, ctx.tags)
```

## Core Concepts

**Slub** provides four main components that work together:

- **Context**: Your custom data structure that threads through the computation
- **Handlers**: Functions that process individual operations and update the context
- **Dispatcher**: Routes operations to their corresponding handlers
- **Interpreter**: Executes JAX computations while applying your handlers via the `reinterpret` decorator

## Built-in Handlers

Slub includes handlers for common JAX operations:

| Handler | Covers | Description |
|---------|--------|-------------|
| `default_primitive_handler` | Primitive ops | Base handler for leaf operations |
| `default_jit_handler` | `jax.jit` | Recursively interprets JIT-compiled functions |
| `default_scan_handler` | `lax.scan` | Handles scan loops with carry/result threading |
| `default_while_handler` | `lax.while_loop` | Supports context growth via initializer/updater |
| `default_cond_handler` | `lax.cond` | Ensures branch contexts have matching structure |

## Control Flow Details

When working with JAX control flow primitives, keep these behaviors in mind:

- **`while_loop`**: The condition function must be pure (no context modification). If the body grows the context, provide an initializer and optionally an updater function.
- **`scan`**: The scan body cannot drop context leaves. You must explicitly choose which parts go into the carry versus the result.
- **`cond`**: All branches must produce identical context tree structures to maintain type consistency.
- **`jit`**: The inner graph is recursively reinterpreted with the same interpreter.

## Example: While Loop with Context Growth

This example shows how to handle a `while_loop` that adds context during execution:

```python
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from functools import partial
from slub.interpreter import Interpreter, Dispatcher, InterpreterContext, reinterpret
from slub.handlers import (
    Uninitialized,
    default_while_handler,
    default_primitive_handler,
)


@partial(
    jax.tree_util.register_dataclass, meta_fields=("tags",), data_fields=("values",)
)
@dataclass(frozen=True)
class Ctx(InterpreterContext):
    tags: tuple[str, ...] = ()
    values: tuple[jax.Array, ...] = ()

    def add_meta(self, tag: str):
        return Ctx(self.tags + (tag,), self.values)

    def add_value(self, v: jax.Array):
        return Ctx(self.tags, self.values + (v,))

    def push(self):
        return self

    def pop(self):
        return self


def sin_handler(interpreter, ctx, eqn, invals):
    # adds metadata and one value
    ctx = ctx.add_meta("sin").add_value(jnp.array(1))
    return default_primitive_handler(interpreter, ctx, eqn, invals)


def initializer(old_ctx, sentinel_ctx):
    # Replace Uninitialized leaves with zeros
    leaves, tree = jax.tree.flatten(sentinel_ctx)
    leaves = [jnp.zeros_like(x) if isinstance(x, Uninitialized) else x for x in leaves]
    return jax.tree.unflatten(tree, leaves)


def updater(old_ctx, new_ctx):
    # Replace old context with new context from loop body
    return new_ctx


def while_with_init(interpreter, ctx, eqn, invals):
    ctx = ctx.add_meta("while").add_value(jnp.array(1))
    return default_while_handler(
        interpreter, ctx, eqn, invals, initializer=initializer, updater=updater
    )


dispatcher = Dispatcher({"while": while_with_init, jax.lax.sin_p: sin_handler})
interpreter = Interpreter(dispatcher=dispatcher)


@partial(reinterpret, interpreter=interpreter)
def run_loop_with_init(x):
    def cond(a):
        return a < 3

    def body(a):
        _ = jnp.sin(a)  # introduces extra context via sin_handler
        return a + 1

    return jax.lax.while_loop(cond, body, 0)


result, out_ctx = run_loop_with_init(Ctx(), jnp.array(0))
```

## Use Cases

Slub is designed for lightweight instrumentation and experimentation:

- **Instrumentation**: Track operations, collect metrics, or monitor computation flow
- **Provenance**: Record the history and lineage of values through a computation
- **Lightweight metrics**: Gather statistics without heavyweight frameworks
- **Research prototyping**: Quickly experiment with custom computation semantics

## Examples

- **`examples/monitoring_pipeline.py`** — Demonstrates primitive handlers, scan, and JIT compilation
- **`notebooks/example.ipynb`** — Interactive notebook with step-by-step examples

## Advanced Features

For advanced usage patterns, see the source code for:

- Custom matching rules for handler dispatch
- Error policies (`RAISE`, `WARN`, `IGNORE`) for mismatched contexts
- Branch combiners for merging contexts from conditional branches

## Development

To set up the development environment:

```bash
uv sync --group dev          # Install dev dependencies
uv run pytest                # Run tests
uvx pre-commit run --all-files  # Run linters and formatters
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) or visit http://www.apache.org/licenses/LICENSE-2.0.

## Version

The project version is defined in `pyproject.toml` under `[project].version`.

## Citation

```bibtex
@software{slub2026,
  title={Slub},
  author={Cusp AI},
  year={2026},
  url={https://github.com/cusp-ai-oss/slub}
}
```
