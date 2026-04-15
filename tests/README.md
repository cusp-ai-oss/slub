# Slub Test Suite

This directory contains a comprehensive test suite for the Slub JAX interpreter framework.

## Structure

### Core Test Files

- **`test_handlers.py`** - Tests for all handler types (Primitive, JIT, Scan)
  - Basic handler execution
  - Composed handlers (nested operations)
  - Complex scenarios like the notebook example

- **`test_interpreter.py`** - Tests for interpreter core functionality
  - Dispatcher functionality
  - Interpreter policies (RAISE, WARN, IGNORE)
  - Context threading through transformations

- **`test_utilities.py`** - Tests for utility functions
  - `split_sequence` function testing
  - Environment utilities
  - Matching rules

- **`test_integration.py`** - End-to-end integration tests
  - Recreation of notebook examples
  - Real-world ML-like scenarios
  - Performance testing
  - Error recovery testing

### Test Infrastructure

- **`conftest.py`** - Shared test fixtures and utilities
  - `TestContext` - Test implementation of `InterpreterContext`
  - `HandlerFactory` - Factory for creating test handlers
  - Helper functions for test setup

## Key Features

### Extensible Test Design

The test suite is designed to be easily extensible for new primitives:

```python
# Example: Adding tests for a new primitive
def test_new_primitive_handler(handler_factory):
    new_handler = handler_factory.create_primitive_handler("new_primitive")
    interpreter = create_test_interpreter({"new_primitive": new_handler})
    # ... rest of test
```

### Context Tracking Validation

All tests verify that context is properly tracked through transformations:

```python
assert "jit" in result_ctx.metadata
assert "scan" in result_ctx.metadata
assert result_ctx.total_metadata_count == 2
```

### Real-World Scenarios

Integration tests include ML-like scenarios:
- RNN-style computations
- Nested control flow
- Large-scale computations

## Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_handlers.py

# Run specific test
uv run pytest tests/test_handlers.py::TestBasicHandlers::test_jit_handler_execution
```

## Extending for New Primitives

The test suite is designed to easily accommodate new JAX primitives like `while`, `cond`, etc.

### Adding a New Primitive Test

1. **Create Handler Factory Method**:
```python
@staticmethod
def create_while_handler(meta_key: str = "while"):
    def handler(interpreter, ctx, eqn, invals):
        ctx = ctx.add_meta(meta_key)
        ctx = ctx.add_value(jnp.array(42))
        return DefaultWhileHandler()(interpreter, ctx, eqn, invals)
    return handler
```

2. **Add Basic Test**:
```python
def test_while_handler_execution(self, handler_factory):
    while_handler = handler_factory.create_while_handler("while")
    interpreter = create_test_interpreter({"while": while_handler})
    # ... test implementation
```

3. **Add Integration Test**:
```python
def test_while_with_other_primitives(self, handler_factory):
    # Test while loops nested with jit, scan, etc.
```

### Test Categories for New Primitives

For each new primitive, consider adding tests for:

1. **Basic functionality** - Does the handler execute and track context?
2. **Composition** - How does it work with existing primitives?
3. **Nesting** - Can it be nested inside JIT/scan and vice versa?
4. **Error handling** - What happens when things go wrong?
5. **Performance** - Does it scale to realistic use cases?

## Test Context Design

The `TestContext` class provides a flexible way to track interpreter state:

```python
@dataclass(frozen=True)
class TestContext(InterpreterContext):
    metadata: tuple[str, ...]  # Track operation types
    parent: TestContext | None  # For push/pop
    level: int                  # Nesting depth
    value: tuple[TracerValue, ...]  # Tracked values
```

This allows tests to verify:
- Which operations were executed (`metadata`)
- Proper nesting behavior (`level`, `parent`)
- Data flow through transformations (`value`)

## Debugging Tests

When tests fail, check:

1. **Context tracking**: Are the expected metadata entries present?
2. **Value preservation**: Are the computed results correct?
3. **Error messages**: Do they indicate the failure mode?
4. **Handler registration**: Are handlers properly registered in the dispatcher?

The test suite includes extensive error cases and debugging information to help identify issues quickly.
