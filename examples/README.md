# Examples

This folder contains runnable examples that showcase Slub in realistic scenarios.

- monitoring_pipeline.py — a small training-like pipeline that combines:
  - `scan` to iterate over micro-batches
  - `cond` to apply a warmup learning rate
  - `while_loop` to retry a step on numerical issues
  - custom context that collects metadata and scalar values

Run:

```bash
uv run python examples/monitoring_pipeline.py
```

You should see output similar to:

```
Output value: <float>
Metadata tags in order: ('cond', 'while', 'scan', 'sin', ...)
Number of collected values: <int>
```

Notes:
- The example uses DefaultWhileHandler with `initializer` and `updater` since the body may introduce extra context.
- The condition function in `while` only reads context; it must not modify it.
- Switch DefaultScanHandler between carry vs constant context threading by changing the `constant_context_threading` flag in the example dispatcher.
