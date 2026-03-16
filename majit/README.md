# majit

**M**eta-tr**A**cing **JIT** compiler framework — or, if you prefer, **Ma**gical **JIT**.

majit is a Rust port of [RPython's JIT infrastructure](https://rpython.readthedocs.io/en/latest/jit/index.html). Given an interpreter written in Rust, majit automatically generates a tracing JIT compiler for it.

## What it does

Write a bytecode interpreter. Annotate it with `#[jit_interp]`. majit does the rest:

```rust
#[jit_interp(state = State, env = Program, ...)]
fn mainloop(program: &Program, state: &mut State, driver: &mut JitDriver<State>) {
    while pc < program.len() {
        jit_merge_point!(driver, program, pc);
        match program[pc] {
            Op::Add => { /* ... */ }
            Op::Jump(target) => {
                pc = target;
                can_enter_jit!(driver, target, state, ...);
                continue;
            }
            // ...
        }
        pc += 1;
    }
}
```

Hot loops are detected, traced, optimized, and compiled to native code via Cranelift. Guard failures fall back to the interpreter transparently.

## Similarities with RPython

majit and the RPython JIT share the same core ideas:

- **Meta-tracing**: traces the interpreter itself, optimizing at the interpreter execution level rather than the bytecode level
- **Guard-based speculation**: records type/value assumptions as guards; deoptimizes to the interpreter on failure
- **Trace → Optimize → Compile → Execute pipeline**
- **8-pass optimizer**: IntBounds, Rewrite, Virtualize, String, Pure, Guard, Simplify, Heap
- **Escape analysis**: eliminates virtual object allocations (NEW → field tracking → force on escape)
- **Resume/blackhole deoptimization**: restores interpreter state on guard failure

## Differences from RPython

### Language model

RPython translates a **restricted Python subset** to C at compile time. The annotator infers types, and the rtyper lowers them to low-level representations. JIT hints (`jit_merge_point`, `promote`, `elidable`, etc.) are inserted directly into Python source.

majit works with **plain Rust**. No type inference is needed (the Rust compiler already handles that), and JIT hints are provided as proc-macro attributes:

| RPython | majit |
|---------|-------|
| `@jit.elidable` | `#[elidable]` |
| `@jit.dont_look_inside` | `#[dont_look_inside]` |
| `jit.JitDriver(greens=[...], reds=[...])` | `#[jit_driver(greens = [...], reds = [...])]` |
| `driver.jit_merge_point(...)` | `jit_merge_point!(driver, ...)` |
| `driver.can_enter_jit(...)` | `can_enter_jit!(driver, ...)` |

### Translation vs analysis

RPython's codewriter **fully translates** RPython source into JitCode (bytecode). All types and control flow are finalized at translation time.

majit **auto-generates and injects trace code at build time**. There are two paths:

1. **`majit-analyze` path (used by pyre-mjit)**: `build.rs` calls `majit_analyze::analyze_multiple()` to parse interpreter sources, then `generate_trace_code()` to produce trace helper functions, writing them to `OUT_DIR/jit_trace_gen.rs`. The main crate pulls them in via `include!`. This path extracts opcode dispatch arms, resolves cross-file trait impls, classifies helpers, and collects type layouts.

2. **`#[jit_interp]` proc-macro path (used by aheui-mjit)**: `build.rs` reads the interpreter source, extracts opcode match arms, and auto-generates a JIT mainloop annotated with `#[jit_interp]`, writing it to `OUT_DIR/jit_mainloop_gen.rs`. The proc macro lowers `while`/`loop` to branch bytecodes, `match` to guard chains, and `for` loops to abort fallback (equivalent to RPython's `@dont_look_inside`).

Both paths achieve **automatic generation and injection**. The remaining difference from RPython is **generality** — how many interpreter shapes and complex CFG patterns can be directly lowered, rather than falling back to opaque residual calls.

### Backend

RPython maintains **6 hand-written assembler backends** for x86, ARM, AArch64, s390x, and PPC (~300K LOC).

majit uses a single [Cranelift](https://cranelift.dev/) backend for all platforms. Cranelift handles ISA-specific code generation, register allocation, and instruction selection, greatly reducing backend code.

### GC

RPython's incminimark GC is deeply integrated with the JIT and uses an lltype-based low-level memory model.

majit's GC (`majit-gc`) implements the same algorithms (nursery + oldgen + incremental marking + card marking) but operates on top of Rust's ownership model. JIT-GC integration hooks (`jit_remember_young_pointer`, `gc_step`, `pin`/`unpin`) are also provided.

### SIMD

RPython's vectorizer targets SSE/AVX directly.

majit uses Cranelift's `I64X2`/`F64X2` SIMD types for platform-independent vectorization, including dependency graph analysis, pack group detection, cost modeling, and instruction scheduling.

## Crate structure

```
majit/                          # facade crate (re-exports all below)
├── majit-ir/                   # IR: OpCode, Type, Value, Descr traits
├── majit-opt/                  # Optimizer: 8-pass pipeline + auxiliaries
├── majit-trace/                # Tracing: hot counter, recorder, warm state
├── majit-codegen/              # Backend abstraction: Backend trait
├── majit-codegen-cranelift/    # Cranelift backend: native code generation
├── majit-meta/                 # Meta-interpreter: JitDriver, MetaInterp, resume
├── majit-gc/                   # GC: nursery, oldgen, incremental, card marking
├── majit-macros/               # Proc macros: #[jit_driver], #[jit_interp], etc.
├── majit-runtime/              # Runtime: jit_merge_point!, can_enter_jit!
├── majit-analyze/              # Static analyzer: source → trace code generation
└── examples/                   # 8 toy interpreters (tlr, tl, tla, tiny2, ...)
```

## Performance

| Interpreter | Program | Interpreter | JIT | Speedup |
|------------|---------|-------------|-----|---------|
| aheuijit | logo.aheui | 6.1s | 0.05s | **110x** |
| pyre | fib(20) | — | correct result | JIT works |

## License

MIT — same as [PyPy/RPython](https://github.com/pypy/pypy/blob/main/LICENSE).
