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

## Core Pieces
Context (your record) · Handlers (per op) · Dispatcher (routing) · Interpreter + reinterpret (execution hook)

## Built-in Handlers
| handler | covers | note |
|---------|--------|------|
| default_primitive_handler | primitive ops | leaf workhorse |
| default_jit_handler | jax.jit | recursively interprets inner jaxpr |
| default_scan_handler | lax.scan | carry vs result threading |
| default_while_handler | lax.while_loop | initializer/updater if body grows ctx |
| default_cond_handler | lax.cond | branch ctx structures must match |

## Example
`examples/monitoring_pipeline.py` (primitive + scan + jit).

## License

Licensed under the Apache License, Version 2.0.

## Control‑flow Gist
while_loop: cond pure; body growth -> initializer (+updater optional)
scan: body cannot drop leaves; choose carry/result
cond: branches -> identical ctx tree
jit: inner graph reinterpreted

```python
import jax
import jax.numpy as jnp
from slub.handlers import DefaultWhileHandler, DefaultPrimitiveHandler, Uninitialized

def sin_handler(interpreter, ctx, eqn, invals):
    # adds metadata and one value
    ctx = ctx.add_meta("sin").add_value(jnp.array(1))
    return DefaultPrimitiveHandler()(interpreter, ctx, eqn, invals)

def initializer(ctx):
    # Replace Uninitialized leaves with zeros
    leaves, tree = jax.tree.flatten(ctx)
    leaves = [jnp.zeros_like(x) if isinstance(x, Uninitialized) else x for x in leaves]
    return jax.tree.unflatten(tree, leaves)

def while_with_init(interpreter, ctx, eqn, invals):
    ctx = ctx.add_meta("while").add_value(jnp.array(1))
    return DefaultWhileHandler(initializer=initializer)(interpreter, ctx, eqn, invals)

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

## Advanced
See source: matching rules, policies (RAISE/WARN/IGNORE), branch combiners.

## Use Cases
Instrumentation · provenance · lightweight metrics · research prototyping

## Examples
Minimal: `examples/monitoring_pipeline.py`. Notebook: `notebooks/example.ipynb`.

## Dev (optional)
Setup: `uv sync --group dev` · Tests: `uv run pytest` · Pre-commit: `uvx pre-commit run --all-files`

## License

Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
## Version

The project version is defined in `pyproject.toml` under `[project].version`.

## Citation
See repository releases if you need to cite; minimal footprint.
