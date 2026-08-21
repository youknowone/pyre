# pyre

[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/2fAUZ49JX3)

**Experimental** - This project is still in development, and not ready for prime time.

**Py**thon **Re**written — a no-GIL Python implementation in Rust, with a meta-tracing JIT compiler ported from PyPy.

## Why pyre?

PyPy proved that a meta-tracing JIT can make Python fast. pyre takes that proven architecture and rebuilds it in Rust — gaining memory safety, no GIL, and a modern toolchain, while keeping the same optimization pipeline that makes PyPy fast.

The key insight: pyre's JIT framework [MaJIT](majit/) handles tracing, optimization, and native code generation. This means the pyre interpreter itself can stay close to a straightforward Rust program that executes Python bytecodes, while MaJIT provides the tracing JIT machinery around it. In the same way that PyPy is "just a Python interpreter" that RPython makes fast, pyre is "just a Rust interpreter" that MaJIT makes fast.

The deeper goal is to **reproduce RPython in Rust**. RPython's real value was never one specific interpreter but the framework that turns an ordinary interpreter into a fast VM — and MaJIT is that reproduction. PyPy is the most complete language ever built on RPython, so porting PyPy is how we prove and complete MaJIT's reproduction of RPython. pyre is the vehicle; a faithful RPython-in-Rust is the destination.

## Status

pyre is under active development. Loop tracing and function inlining work, and the JIT fires on integer-, float-, and exception-heavy loops alike. On the CI benchmark set the default backend runs nine of the ten programs at 2.0x of PyPy or better, and `float_loop` faster than PyPy; `fannkuch` is the widest remaining gap at 2.5x. Many Python features are not yet implemented.

## Benchmarks

Copied from the `pyre/check.py (ubuntu-24.04)` job of [CI run 32465640260](https://github.com/youknowone/pyre/actions/runs/32465640260), a single-core GitHub Actions `ubuntu-24.04` runner. The figure in parentheses is check.py's ratio against PyPy.

| Benchmark | CPython 3.14 | PyPy 7.3 | dynasm | cranelift | wasm |
|-----------|--------------|----------|--------|-----------|------|
| int_loop | – | 0.32s | 0.43s (1.0x) | 0.58s (1.5x) | 0.47s (1.4x) |
| float_loop | – | 0.42s | 0.31s (0.5x) | 0.37s (0.7x) | 0.25s (0.5x) |
| fib_loop | 0.20s | 0.12s | 0.27s (1.6x) | 0.27s (1.5x) | 0.26s (2.1x) |
| inline_helper | – | 0.22s | 0.35s (1.2x) | 0.35s (1.2x) | 0.41s (1.7x) |
| fib_recursive | 1.92s | 0.31s | 0.72s (2.0x) | 1.03s (3.1x) | 2.38s (7.8x) |
| nested_loop | – | 0.30s | 0.51s (1.4x) | 0.74s (2.2x) | 0.82s (2.7x) |
| raise_catch | – | 0.82s | 0.93s (1.0x) | 1.65s (1.9x) | 1.50s (1.8x) |
| spectral_norm | – | 0.13s | 0.31s (1.7x) | 0.42s (2.6x) | 0.36s (2.6x) |
| nbody | – | 0.30s | 0.60s (1.7x) | 0.77s (2.3x) | 0.83s (2.7x) |
| fannkuch | 2.71s | 0.34s | 0.94s (2.5x) | 1.77s (5.1x) | 1.87s (5.6x) |

`dynasm` is the default backend of the `pyre` binary; `cranelift` and `wasm` are the other two MaJIT code generators.

Reading the table: the printed times are user CPU time — `check.py`'s `run_timed` reports `ru_utime` on Unix and `TotalUserTime` on Windows — and include interpreter startup, which on that runner was 0.010s for CPython, 0.013s for PyPy, 0.109s for dynasm, 0.106s for cranelift and 0.046s for wasm. The ratios divide execution-only times, so they are not the quotient of the printed columns — that startup difference is why `int_loop` prints a third more than PyPy's time yet ratios at 1.0x. A `–` in the CPython column means check.py ran no CPython reference for that benchmark, not that it timed out: it measures one only where a vs-CPython gate is configured, and `--full` measures it everywhere. Absolute times are runner-dependent and only comparable within one table; the ratios are the part that carries across runs.

On execution-only time the default backend is at 2.0x of PyPy or better on nine of the ten. `float_loop` beats PyPy on all three backends, and `int_loop` and `raise_catch` reach parity on dynasm. `fannkuch` is the widest gap. Cranelift trails dynasm on eight of the ten — widest on `fannkuch` (~2.0x) and `raise_catch` (~1.9x) — and matches it only on `fib_loop` and `inline_helper`. Where CPython was measured, pyre runs `fannkuch` ~3.3x and `fib_recursive` ~3.1x faster than it, and `fib_loop` ~1.2x.

Run `python pyre/check.py` to reproduce all benchmarks with CPython / PyPy / pyre comparison on your machine. If the release backend binaries are already built, pass `--no-build` to skip the Cargo build phase.

## Installation

### Homebrew

```sh
brew install youknowone/tap/pyrex
```

The formula lives in the [youknowone/homebrew-tap](https://github.com/youknowone/homebrew-tap/tree/main/Formula).

### Prebuilt binaries

Download a prebuilt binary from the [GitHub releases page](https://github.com/youknowone/pyre/releases).

### Cargo

```sh
cargo install pyrex
```

## Building from source

```sh
python3 scripts/install-charon.py
python3 scripts/extract-llbc.py
cargo build --release -p pyrex
./target/release/pyre script.py
```

`install-charon.py` installs the pinned Charon version used to produce the
LLBC artifacts; `extract-llbc.py` generates the artifacts under `build/llbc`
that MaJIT needs during its build.

## How it works

pyre follows PyPy's meta-tracing approach:

1. The **interpreter** (`pyre-interpreter`) executes Python bytecodes normally.
2. When a loop or function becomes hot, **MaJIT** records the interpreter's execution as a linear trace of IR operations.
3. The trace passes through an **8-pass optimizer** — the same pipeline as PyPy: IntBounds, Rewrite, Virtualize, String, Pure, Guard, Simplify, Heap.
4. The optimized IR is compiled by one of three MaJIT backends: `dynasm` (the default; x86-64 and aarch64) and Cranelift emit **native machine code**; the third emits **WebAssembly**, which `pyre-wasm-runner` executes under wasmtime by default, or under wasmi.
5. Subsequent executions of that path run the compiled code directly. Guard failures fall back to the interpreter.

### Function inlining

During loop tracing, pyre traces *through* function call boundaries. A call to `add(a, b)` in the loop body becomes `IntAddOvf(a_raw, b_raw)` in the compiled trace — no function call overhead, no frame allocation.

### no-GIL

pyre has no Global Interpreter Lock. RPython/PyPy features that depend on the GIL have no equivalent trigger path in pyre. The API surfaces are kept for naming parity with the original codebase but have no production call sites.

## Crate structure

```
pyre/
├── pyre-object      # Python object types (W_IntObject, W_FloatObject, W_ListObject, ...)
├── pyre-macros      # Proc macros for builtin module declarations (@unwrap_spec equivalent)
├── pyre-native      # Native library backends, kept outside the LLBC extraction
├── pyre-interpreter # Object space, interpreter frame, eval loop, opcode dispatch, bytecode
├── pyre-module      # Optional builtin modules
├── pyre-sandbox     # RPython-style sandbox protocol, virtual filesystem and controller
├── pyre-jit         # JIT compiler integration for the interpreter
├── pyre-jit-trace   # Trace-time JIT — MIFrame and tracing logic
├── pyre-wasm        # The interpreter compiled to WebAssembly
├── pyre-wasm-runner # Native wasmtime host satisfying the JIT host-import contract
├── pyre-wasm-test   # Interpreter-only smoke binary run inside the wasm sandbox
└── pyrex            # Executable entry point (builds the `pyre` binary)
```

## Relationship to PyPy

pyre is a structural port of PyPy's interpreter (`pypy/interpreter/` and `pypy/objspace/`). Every module, type, and function in the original Python codebase exists in the Rust port under the same name at the same relative location — only snake_case conversion is applied to method names. This naming parity makes it possible to read the PyPy source alongside pyre and see exactly what each piece corresponds to.

## Key differences from PyPy

- **Rust-source translation through MaJIT.** RPython translates a live Python
  program image through flowspace, annotator, rtyper, and codewriter. Pyre
  extracts Rust crates to Charon `.ullbc` artifacts and runs their graphs
  through MaJIT's corresponding
  `front → flowspace → annotator → rtyper → codewriter` pipeline at
  `cargo build` time. The explicit `eval::eval_loop_jit` portal seeds the
  ordinary graph closure that becomes assembled JitCodes.
- **Proc macros instead of decorators.** `@jit.elidable` becomes `#[elidable]`, `driver.jit_merge_point(...)` becomes `jit_merge_point!`. Same semantics, Rust syntax.
- **No GIL.** pyre is free-threaded from day one. GIL-dependent code paths in PyPy (heapcache resets on GIL release, `release_gil` effect info, etc.) simply don't exist.
- **Python 3.14, not 2.7.** PyPy's main branch targets Python 2.7/3.10. pyre targets CPython 3.14 bytecodes directly, using RustPython's compiler frontend.

## Relationship to MaJIT

[MaJIT](majit/) (**M**eta-tr**A**cing **JIT**) is a standalone Rust port of RPython's JIT infrastructure. It is a general-purpose framework for Rust bytecode interpreters that integrate with its tracing interface. pyre is MaJIT's primary consumer, but MaJIT has no dependency on pyre.

## Roadmap

What's next, roughly in priority order:

- **Trace exit cost** — `fannkuch` at 2.5x of PyPy is the largest remaining single-benchmark gap; most of what is left is paid on guard failure, transferring state into bridges, rather than inside the compiled loop.
- **Cranelift backend parity** — cranelift trails the default dynasm backend on eight of the ten benchmarks, by up to ~2.0x.
- **Broader JIT coverage** — float and exception JIT now fire on the hot-loop benchmarks; extend that coverage to more of the language.
- **More Python built-ins** — str methods, dict operations, list comprehensions, generators.
- **Multi-threaded execution** — the no-GIL foundation is there; actual parallel thread scheduling is not.
- **CPython C extension compatibility** — long-term goal, likely via HPy or similar ABI layer.

## Name

**pyrex** = **pyre** e**x**ecutable. The `pyrex` crate builds the `pyre` command-line binary.

## License

MIT — same as [PyPy](https://github.com/pypy/pypy/blob/main/LICENSE).
