# majit

**M**eta-tr**A**cing **JIT** compiler framework — or, if you prefer, **Ma**gical **JIT**.

majit is a Rust port of [RPython's JIT infrastructure](https://rpython.readthedocs.io/en/latest/jit/index.html). Given an interpreter written in Rust, majit generates a tracing JIT compiler for it.

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

Hot loops are detected, traced, optimized, and compiled to native code. Guard failures fall back to the interpreter transparently.

## Similarities with RPython

majit and the RPython JIT share the same core ideas — and, per the project's parity rule, the same module names and data structures:

- **Meta-tracing**: traces the interpreter itself, optimizing at the interpreter execution level rather than the bytecode level
- **Guard-based speculation**: records type/value assumptions as guards; deoptimizes to the interpreter on failure
- **Trace → Optimize → Compile → Execute pipeline**
- **optimizeopt pass pipeline**: IntBounds, Rewrite, Virtualize, String, Pure, Guard, Simplify, Heap (plus generated rewrite rules in `ruleopt`)
- **Escape analysis**: eliminates virtual object allocations (NEW → field tracking → force on escape)
- **Resume/blackhole deoptimization**: restores interpreter state on guard failure
- **Hint vocabulary**: `promote`, greens/reds, `elidable`, `dont_look_inside`, virtualizables, quasi-immutable fields — same names, same need-oriented placement

## Differences from RPython

### Language model

RPython translates a **restricted Python subset** to C at compile time. The annotator infers types, and the rtyper lowers them to low-level representations. JIT hints (`jit_merge_point`, `promote`, `elidable`, etc.) are inserted directly into Python source.

majit works with **plain Rust**. Type recovery is not needed (the Rust compiler already did it), and JIT hints are proc-macro attributes:

| RPython | majit |
|---------|-------|
| `@jit.elidable` | `#[elidable]` |
| `@jit.dont_look_inside` | `#[dont_look_inside]` |
| `jit.JitDriver(greens=[...], reds=[...])` | `#[jit_driver(greens = [...], reds = [...])]` |
| `driver.jit_merge_point(...)` | `jit_merge_point!(driver, ...)` |
| `driver.can_enter_jit(...)` | `can_enter_jit!(driver, ...)` |

### Translation: live image vs extracted artifacts

RPython analyses a **live program image** (full Python runs as a preprocessor, then flowspace/annotator/rtyper analyse the loaded functions) and its codewriter translates the interpreter into JitCode bytecode that the meta-interpreter executes.

majit runs the **same pipeline at `cargo build` time over extracted artifacts**. There are two front-ends:

1. **Charon LLBC path (used by pyre)**: the interpreter crates are extracted
   to `.ullbc` low-level IR with [Charon](https://github.com/AeneasVerif/charon)
   (`scripts/install-charon.py`, `scripts/extract-llbc.py`), and
   **majit-translate** consumes them through the RPython pipeline shape —
   `front` → `flowspace/` → `annotator/` → `rtyper/` → `codewriter/` — to
   produce JitCode. `majit-charon-reader` is the input layer. Like RPython's
   frozen image, extraction can go stale: source changes are invisible until
   re-extraction (fingerprint skipping handles the common case).

   Consumers configure one or more `JitDriverSpec` values with exact,
   qualified portal `CallPath`s. MaJIT registers every driver before graph
   discovery, then `make_jitcodes` compiles the ordinary call-graph closure
   reachable from those portals. Interpreter dispatch functions and opcode
   helpers are normal graphs in that closure; the LLBC path does not construct
   a parallel opcode-arm table or synthetic dispatch namespace.

   During `pyre-jit.ullbc` extraction, the extraction driver automatically sets
   the internal `MAJIT_LLBC_EXTRACTION=1` build mode to break the
   `pyre-jit → pyre-jit-trace → pyre-jit.ullbc` bootstrap cycle. That mode
   emits compile-only placeholder artifacts and is not a supported runtime
   configuration; users should run `scripts/extract-llbc.py`, not set the
   variable themselves. The next normal Cargo build regenerates real artifacts
   from the completed LLBC set.

2. **`#[jit_interp]` proc-macro path (used by aheui-mjit and `examples/`)**:
   `build.rs` reads the interpreter source, extracts opcode match arms, and
   generates a JIT mainloop annotated with `#[jit_interp]`. The proc macro
   lowers `while`/`loop` to branch bytecodes, `match` to guard chains, and
   unsupported shapes to residual-call fallback (the `@dont_look_inside`
   equivalent).

The remaining gap to RPython is **generality**: how many interpreter shapes
lower directly instead of falling back to opaque residual calls. Fallbacks
are tracked by a census, not accepted silently.

### Backend

RPython maintains **hand-written assembler backends** for x86, ARM, AArch64, s390x, and PPC (~300K LOC).

majit keeps three thin backends behind one `Backend` trait (`majit-backend`, the `AbstractCPU` analog) instead of hand-writing a full backend per ISA:

- **majit-backend-dynasm** — the current primary backend: direct machine-code emission via dynasm-rs (low compile latency)
- **majit-backend-cranelift** — portable option that delegates instruction selection and register allocation downward ([Cranelift](https://cranelift.dev/))
- **majit-backend-wasm** — emits WebAssembly trace modules (browser via wasm-bindgen, or native embedders like wasmi/wasmtime)

### GC

RPython's incminimark GC is deeply integrated with the JIT and uses an lltype-based low-level memory model.

majit's GC (`majit-gc`) ports the same lineage (nursery + oldgen + incremental marking + card marking) on top of Rust's ownership model, with the JIT-GC integration hooks (`jit_remember_young_pointer`, `gc_step`, `pin`/`unpin`) and shadow-stack root finding.

### SIMD

RPython's vectorizer targets SSE/AVX directly.

majit uses Cranelift's `I64X2`/`F64X2` SIMD types for platform-independent vectorization, including dependency graph analysis, pack group detection, cost modeling, and instruction scheduling.

## Crate structure

```
majit/                        # facade crate (re-exports the crates below)
├── majit-ir/                 # IR: resoperation model, Descr, effectinfo, intbounds, resume data
├── majit-trace/              # Tracing engine: hot counters, recorder, warm state
├── majit-metainterp/         # Meta-interpreter: pyjitpl, optimizeopt/ruleopt, resume,
│                             #   blackhole, warmspot/warmstate, virtualizable, heapcache
├── majit-translate/          # Build-time translation pipeline:
│                             #   front → flowspace → annotator → rtyper → codewriter
├── majit-charon-reader/      # Parser for Charon .llbc/.ullbc JSON (majit-translate input)
├── majit-backend/            # Backend trait (AbstractCPU parity)
├── majit-backend-cranelift/  # Cranelift code generation
├── majit-backend-dynasm/     # dynasm-rs direct machine-code backend
├── majit-backend-wasm/       # WebAssembly backend (wasm-encoder)
├── majit-gc/                 # GC: nursery + oldgen + incremental + card marking
├── majit-macros/             # Proc macros: #[jit_driver], #[elidable], #[dont_look_inside], …
├── charon-corpus/            # Checked-in LLBC corpus for translate tests
└── examples/                 # 11 toy interpreters (tl, tla, tlc, tlr, tiny2, tiny3,
                              #   tinyframe, braininterp, calc, dualtape, i64env)
```

Consumers today: **pyre** (`pyre-jit`, `pyre-jit-trace` — the Charon path),
**aheui-mjit** and the in-tree `examples/` (the proc-macro path). majit never
depends on pyre; the multi-consumer setup is deliberate — it is the proof of
generality, the role RPython's non-Python interpreters played.

## Verification

- `cargo test` per crate (run with a backend feature, e.g. `--features dynasm`).
- The dual-backend synthetic suite `python3 ./pyre/check.py` runs every pyre
  fixture under both native backends and byte-compares output; it is the
  acceptance gate for JIT changes, alongside the benchmark suite.

## License

MIT — same as [PyPy/RPython](https://github.com/pypy/pypy/blob/main/LICENSE).
