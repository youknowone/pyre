# pyre

**Py**thon **Re**written — a no-GIL Python implementation in Rust, with a meta-tracing JIT compiler ported from PyPy.

## Why pyre?

PyPy proved that a meta-tracing JIT can make Python fast. pyre takes that proven architecture and rebuilds it in Rust — gaining memory safety, no GIL, and a modern toolchain, while keeping the same optimization pipeline that makes PyPy fast.

The key insight: pyre's JIT framework [majit](../majit/) handles all the tracing, optimization, and native code generation. This means the pyre interpreter itself is just a straightforward Rust program that executes Python bytecodes — majit turns it into a JIT compiler automatically. In the same way that PyPy is "just a Python interpreter" that RPython makes fast, pyre is "just a Rust interpreter" that majit makes fast.

## Status

pyre is under active development. Loop tracing and function inlining work. Integer-heavy benchmarks already match or approach PyPy performance. Many Python features are not yet implemented.

## Benchmarks

Measured on Apple M-series, single core:

| Benchmark | pyre (JIT) | PyPy 7.3 | CPython 3.14 | vs CPython |
|-----------|-----------|----------|--------------|------------|
| fib_loop(100k) | 0.06s | 0.08s | 0.14s | **2.3x** |
| inline_helper(1M) | 0.11s | 0.02s | 0.21s | **1.9x** |

pyre matches PyPy on tight integer loops and is consistently faster than CPython where the JIT fires.

Run `pyre/check.sh` to reproduce all benchmarks with CPython / PyPy / pyre comparison on your machine.

## Building & running

```sh
cargo build --release -p pyrex
./target/release/pyre script.py
```

## How it works

pyre follows PyPy's meta-tracing approach:

1. The **interpreter** (`pyre-interpreter`) executes Python bytecodes normally.
2. When a loop or function becomes hot, **majit** records the interpreter's execution as a linear trace of IR operations.
3. The trace passes through an **8-pass optimizer** — the same pipeline as PyPy: IntBounds, Rewrite, Virtualize, String, Pure, Guard, Simplify, Heap.
4. The optimized IR is compiled to **native machine code** via Cranelift.
5. Subsequent executions of that path run the compiled code directly. Guard failures fall back to the interpreter.

### Function inlining

During loop tracing, pyre traces *through* function call boundaries. A call to `add(a, b)` in the loop body becomes `IntAddOvf(a_raw, b_raw)` in the compiled trace — no function call overhead, no frame allocation.

### no-GIL

pyre has no Global Interpreter Lock. RPython/PyPy features that depend on the GIL have no equivalent trigger path in pyre. The API surfaces are kept for naming parity with the original codebase but have no production call sites.

## Crate structure

```
pyre/
├── pyre-object      # Python object types (W_IntObject, W_FloatObject, W_ListObject, ...)
├── pyre-bytecode    # Bytecode definitions (re-exports RustPython compiler-core)
├── pyre-interpreter # Object space, interpreter frame, eval loop, opcode dispatch
├── pyre-jit         # JIT integration — trace recording, call bridges, inlining
└── pyrex            # Executable entry point (builds the `pyre` binary)
```

## Relationship to PyPy

pyre is a structural port of PyPy's interpreter (`pypy/interpreter/` and `pypy/objspace/`). Every module, type, and function in the original Python codebase exists in the Rust port under the same name at the same relative location — only snake_case conversion is applied to method names. This naming parity makes it possible to read the PyPy source alongside pyre and see exactly what each piece corresponds to.

## Relationship to majit

[majit](../majit/) (**M**eta-tr**A**cing **JIT**) is a standalone Rust port of RPython's JIT infrastructure. It is a general-purpose framework — any Rust bytecode interpreter annotated with `#[jit_interp]` gets a tracing JIT for free. pyre is majit's primary consumer, but majit has no dependency on pyre.

## Roadmap

What's next, roughly in priority order:

- **Float-heavy JIT paths** — nbody and fannkuch run correctly but don't yet JIT-compile float operations; closing this gap is the biggest performance unlock remaining.
- **More Python built-ins** — str methods, dict operations, list comprehensions, generators.
- **Exception-heavy JIT** — raise/catch inside traced loops (raise_catch benchmark works interpreted, JIT bridge in progress).
- **Multi-threaded execution** — the no-GIL foundation is there; actual parallel thread scheduling is not.
- **CPython C extension compatibility** — long-term goal, likely via HPy or similar ABI layer.

## Name

**pyrex** = **pyre** e**x**ecutable. The `pyrex` crate builds the `pyre` command-line binary.

## License

MIT — same as [PyPy](https://github.com/pypy/pypy/blob/main/LICENSE).
