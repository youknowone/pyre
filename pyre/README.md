# pyre

**Python Yet Reforged Entirely**

A Python implementation in Rust with a meta-tracing JIT compiler, ported from PyPy/RPython.

## What is pyre?

pyre is a from-scratch Python interpreter written in Rust, equipped with a
meta-tracing JIT framework ([majit](../majit/)) that ports PyPy's optimization
pipeline. It compiles hot loops and frequently-called functions into native
machine code at runtime.

## Crate structure

| Crate | Description |
|-------|-------------|
| `pyre-object` | Python object types (W_IntObject, W_FloatObject, W_ListObject, ...) |
| `pyre-runtime` | Opcode dispatch, shared handlers, runtime operations |
| `pyre-objspace` | Object space — type coercion, truth values, arithmetic |
| `pyre-bytecode` | Re-export of RustPython's compiler-core bytecode definitions |
| `pyre-interp` | Interpreter frame, eval loop (JIT-free) |
| `pyre-jit` | JIT integration — trace recording, inline tracing, call bridges |
| **`pyrex`** | **Executable entry point** — builds the `pyre` binary |

## Building

```sh
cargo build --release -p pyrex
```

This produces `target/release/pyre`.

## Running

```sh
./target/release/pyre script.py
```

## Benchmarks

Measured on Apple M-series, single core:

| Benchmark | pyre (JIT) | PyPy 7.3 | CPython 3.14 |
|-----------|-----------|----------|--------------|
| fib_loop(100k) | 0.06s | 0.08s | 0.14s |
| inline_helper(1M) | 0.11s | 0.02s | 0.21s |

- **fib_loop**: pyre matches PyPy, 2.3x faster than CPython
- **inline_helper**: pyre 2x faster than CPython (function inlining active)

## Architecture

pyre follows PyPy's meta-tracing approach:

1. **Interpreter** (`pyre-interp`): executes Python bytecodes
2. **Trace recorder** (`majit-trace`): records hot paths as IR operations
3. **Optimizer** (`majit-opt`): 8-pass pipeline (IntBounds, Rewrite, Virtualize, String, Pure, Guard, Simplify, Heap)
4. **Backend** (`majit-codegen-cranelift`): compiles optimized IR to native code via Cranelift

### Function inlining

During loop tracing, pyre traces *through* function call boundaries
(PyPy's `_interpret`/`ChangeFrame` pattern). A call to `add(a, b)` in
the loop body becomes `IntAddOvf(a_raw, b_raw)` in the compiled trace —
no function call overhead, no frame allocation.

## Name

**pyrex** = **pyre** e**x**ecutable. The `pyrex` crate builds the `pyre` command-line binary.

## License

Same as PyPy (MIT).
