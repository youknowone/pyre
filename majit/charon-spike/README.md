# Charon Step 0 Spike

Tracking artifact for [issue #97](https://github.com/youknowone/pyre/issues/97)
Step 0 — _extraction spike_.

The goal of Step 0 is **not** to start the migration. It is to prove that
Charon can extract a usable IR from our crates, and to record what the
extracted `.llbc` actually contains before any production code is written
against it.

## Layout

```
charon-spike/
├── README.md              # this file
├── corpus/                # hand-written micro-crate (4 representative shapes)
│   ├── Cargo.toml
│   └── src/lib.rs
├── corpus.ullbc           # extracted ULLBC for the corpus crate (checked in)
├── inspect_llbc.py        # JSON dumper used to produce the findings below
└── prototype/             # Step 0.4: minimal MIR → FunctionGraph lowering
    ├── README.md          # mapping table + structural deltas
    ├── Cargo.toml         # isolated workspace
    ├── src/
    │   ├── main.rs        # CLI: charon-spike-lower
    │   ├── ullbc.rs       # minimal serde decode
    │   ├── graph.rs       # FunctionGraph-shaped target IR
    │   └── lower.rs       # ULLBC → graph translation
    ├── expected/          # checked-in canonical snapshots
    └── compare.sh         # regenerate + diff
```

## Setup notes (Charon install)

Use the canonical fetcher (Step 2.1):

```sh
scripts/install-charon.sh             # installs to ./build/charon/
./build/charon/charon toolchain-path  # one-time nightly install (~1 min)
```

The script pins `CHARON_VERSION_DEFAULT="nightly-2026.05.24"` and stamps
the installed version so re-runs are idempotent. Override with
`CHARON_VERSION=nightly-YYYY.MM.DD scripts/install-charon.sh` to bump.

Charon itself is `0.1.196`. It internally pins Rust toolchain
`nightly-2026-02-07` (`rustc-dev`, `llvm-tools-preview`, `rust-src`,
`miri` components); `charon toolchain-path` auto-installs the
toolchain on first run via rustup. There is no formal stable release
tag — every release is a nightly tag.

The extraction toolchain is **separate from** the toolchain our codebase
compiles with. The downstream stable-Rust consumer never needs the pinned
nightly; it only needs the `.llbc` JSON.

## Reproducing

```sh
# 0. install Charon (once)
../../scripts/install-charon.sh

# 1. extract corpus
cd corpus
../../../build/charon/charon cargo --ullbc --dest-file ../corpus.ullbc

# 2. inspect
cd ..
python3 inspect_llbc.py corpus.ullbc                  # summary
python3 inspect_llbc.py corpus.ullbc desugar_mix      # detailed BB dump

# 3. run the prototype lowering and diff against snapshots
cd prototype && ./compare.sh
```

## Corpus

Four functions from `corpus/src/lib.rs`, chosen to cover the shapes called
out in issue #97 Step 0:

| Function            | Shape                                       | ULLBC BBs |
|---------------------|---------------------------------------------|-----------|
| `straight_line_add` | straight-line arithmetic                    | 5         |
| `branch_loop_sum`   | `for` loop + `if/else` branch               | 13        |
| `strategy_len`      | enum `match` (dict-strategy stand-in)       | 5         |
| `parse_one`         | `match` with guards, internal helper for #4 | 9         |
| `desugar_mix`       | `?` + `for` + `match` + `break`             | 22        |

## Findings

### 1. `.llbc` top-level shape

```json
{
  "charon_version": "0.1.196",
  "has_errors": false,
  "translated": {
    "crate_name": "...",
    "options": {...},
    "target_information": {...},
    "files": [...],
    "item_names": [...],
    "type_decls":  [...],
    "fun_decls":   [...],
    "global_decls":[...],
    "trait_decls": [...],
    "trait_impls": [...],
    "ordered_decls": {...}
  }
}
```

- Types are **deduplicated globally** via `{"Deduplicated": <id>}` references
  into a parallel pool. Inline forms appear as
  `{"HashConsedValue": [<id>, <ty>]}` for the first occurrence.
- `fun_decls` includes **opaque references** to functions defined in other
  crates (their bodies are `null`). In the corpus crate, **5 of 213
  `fun_decls` carry our local bodies**; the rest are pulled in by name from
  `core` because they appear in call positions (iterator/Try/etc.).

### 2. Per-function body — ULLBC (basic-block CFG)

`charon cargo --ullbc` produces the unstructured form, which is the
direct analog of MIR and the right shape for issue #97's BFS driver. The
default `charon cargo` output reconstructs control-flow into a structured
nested AST and is **not** what we want.

ULLBC body shape:

```json
"body": {
  "Unstructured": {
    "span":             { "data": { "file_id": ..., "beg": {...}, "end": {...}}},
    "bound_body_regions": 0,
    "locals": {
      "arg_count": <n>,
      "locals": [{ "index": i, "name": null|"s", "span": {...}, "ty": ... }]
    },
    "body": [<BasicBlock>...]
  }
}
```

`BasicBlock` shape:

```json
{
  "statements":  [<Statement>...],
  "terminator":  { "span": ..., "kind": <TermKind>, "comments_before": [...] },
  "span":        { ... }
}
```

#### Locals

- Slot `0` = return value place; slots `1..arg_count` = arguments;
  the rest = compiler-introduced temporaries.
- Each local carries: `index`, optional `name` (preserved for user-named
  bindings, `null` for compiler-introduced temps), a precise source `span`,
  and a (deduplicated) `ty`.
- This means our lowering driver can recover **user-meaningful names** for
  bindings — the AST front-end currently does this from the `syn` tree.
- It also means **rustc-introduced temporaries are visible**, including the
  tuple-typed temp used by `AddChecked` (see §4 below).

#### Statements

Observed statement `kind` variants in the corpus:

| Kind                           | What it is                                              |
|--------------------------------|----------------------------------------------------------|
| `StorageLive(local_idx)`       | mark local slot alive                                   |
| `StorageDead(local_idx)`       | mark local slot dead                                    |
| `Assign(Place, Rvalue)`        | the main building block                                 |
| `Assert{...}`                  | inline assertion (e.g. overflow check)                  |

`Place` is `{ "kind": Local(i) | Projection(Place, ProjectionElem), "ty": ... }`
where `ProjectionElem` includes `Field(VariantId, FieldId)`, deref, indexing,
etc. (none of which the AST front-end ever sees pre-resolved).

`Rvalue` observed forms include `Use(Operand)`, `BinaryOp(Op, Operand, Operand)`,
`UnaryOp`, `Aggregate`, `Ref`, `Cast`, `Discriminant(Place)`, `Len(Place)`,
etc. `Operand` is `Copy(Place) | Move(Place) | Constant(ConstantExpr)`.

#### Terminators

Observed terminator `kind` variants in the corpus:

| Kind                                       | Successors             |
|--------------------------------------------|-------------------------|
| `Return`                                   | (none)                  |
| `UnwindResume`                             | (none, ABI exit)        |
| `Abort`                                    | (none)                  |
| `Goto { target }`                          | `target`                |
| `Switch { discr, targets: If(then, else) }`| `then`, `else`          |
| `Switch { discr, targets: SwitchInt(ty, cases: [(scalar, bb)], default) }` | each case + default |
| `Call { call, target, on_unwind }`         | `target` (success), `on_unwind` (panic edge) |
| `Assert { assert, target, on_unwind }`     | `target`, `on_unwind`   |
| `Drop { ..., target, on_unwind }`          | `target`, `on_unwind`   |

Every fallible terminator (`Call`, `Assert`, `Drop`) carries an
**explicit `on_unwind` edge** to the unwind landing pad. Issue #97 called
out panic/unwind information as something MIR provides that the AST
front-end has to reconstruct — this is where it lives.

### 3. What Charon resolves that the AST front-end re-derives

| Information                | `syn` AST front-end                   | Charon ULLBC                                |
|----------------------------|----------------------------------------|-----------------------------------------------|
| Concrete `Ty` per local    | inferred via `annotator`/`rtyper`      | already attached to every local + place      |
| Trait method resolution    | resolved via shims                     | `Call.func.Regular.kind.Fun.Regular(<id>)` is a direct `fun_decls` index; `trait_refs` carries instantiation |
| `?` desugaring             | manually unrolled (`front/raise.rs`)   | rewritten into `Try::branch` + `FromResidual::from_residual` calls + Switch on `ControlFlow` |
| `for` desugaring           | manually unrolled                      | rewritten into `IntoIterator::into_iter` + loop over `Iterator::next` + Switch on `Option` |
| Generic instantiation      | partial (best-effort)                  | full, with `generics: { regions, types, const_generics, trait_refs }` |
| Source spans               | from `syn` tokens                      | from rustc, file_id + beg/end line/col       |
| Overflow checks            | not visible                            | `BinaryOp("AddChecked", ...)` + paired `Assert.check_kind.Overflow(...)` (see §4) |
| Drop / unwind edges        | not visible                            | `Drop` and unwind terminators are first-class |

### 4. Surprises and pitfalls

- **Overflow-checked arithmetic is in the ULLBC.** `a + b` in debug mode
  lowers to `Assign(t, BinaryOp("AddChecked", a, b))` followed by
  `Assert(!t.1, on_failure: Panic, check_kind: Overflow(Add Wrap, a, b))`
  and a `Assign(result, t.0)`. Our lowering driver has to either honor
  these (emit overflow checks) or strip them (treating `AddChecked` as
  `Add Wrap` and removing the paired `Assert`). For interpreter-style
  hot paths the latter is almost certainly what we want, but the
  decision must be explicit.
- **`Structured` vs `Unstructured`.** Charon's default (`charon cargo`)
  reconstructs the CFG into a nested-AST `Structured` form. That form
  has the advantage that loops and `if/else` are explicit syntactic
  constructs (closer to `syn`), but it is **not** what mirrors
  `flowcontext.py`'s BFS over basic blocks. For this project we should
  always use `--ullbc`. See `corpus.ullbc` for the artifact.
- **Cargo features matter.** Charon invokes `cargo build` under the
  hood, so feature flags affect what compiles. `pyre-interpreter`
  pulls in `majit-metainterp`, which has an explicit `compile_error!`
  unless `cranelift` or `dynasm` is enabled. The right invocation is
  `charon cargo --ullbc -- --features cranelift` (or `dynasm`).
- **`thread_local!` accessors translate as `Error` bodies.** Charon
  emits a per-statement error (`"charon does not support thread local
  references"`) but the surrounding function still extracts. In
  `pyre-interpreter` this produced 70 `Error`-bodied accessor
  closures out of 8,940 with-body decls; none of the interpreter's
  own functions failed. Downstream consumers must treat
  thread-local reads as opaque ops.
- **`has_errors: true` is set whenever any item failed**, even if it
  was a generated accessor. So `has_errors` alone isn't a hard gate —
  the per-item `body: { "Error": {...} }` walk is what matters.

### 5. Real-crate extraction smoke tests

These were **not** checked into the repo (the `.ullbc` files are too
large to commit), but the runs are reproducible from the commands
below. The numbers were collected on the `charon` branch at the head
of this spike.

#### `pyre-object`

```
charon cargo --ullbc --dest-file /tmp/pyre-object.ullbc
```

Result: **22.7 MB ULLBC. Compiled cleanly.**
- 3,733 `fun_decls` total; 2,422 with body slot; 28 `Error` bodies
  (all `thread_local!` accessors).
- 1,861 `pyre_object::*` functions with bodies extracted.
- 500 `core`/`std`/`alloc` bodies pulled in by reference.

#### `pyre-interpreter`

```
charon cargo --ullbc --dest-file /tmp/pyre-interpreter.ullbc \
  -- --features cranelift
```

Result: **133 MB ULLBC. Compiled cleanly with `cranelift` feature.**
- 11,728 `fun_decls` total; 8,940 with body slot; 70 `Error` bodies
  (all `thread_local!`).
- 6,448 `pyre_interpreter::*` functions extracted.
- Pulled-in deps with bodies: `core` 1,314, `pyre_object` 371,
  `std` 267, `alloc` 165, `rustpython_compiler_core` 93,
  `malachite_bigint` 90, `pymath` 54, `rustpython_sre_engine` 54,
  `libc` 33, others ≤ 20.
- BB-count distribution (local crate only):
  median 5, p90 36, p99 136, **max 2,225**
  (`pyre_interpreter::pyopcode::execute_opcode_step`).
- Terminator distribution (all extracted fns):
  Call 25,920, Drop 14,894, Abort 9,417, Goto 8,043, Return 7,169,
  Switch 6,218, UnwindResume 5,749, Assert 1,666.
- Only **7 `type_decls` carry `DynTrait`** — much lower than the raw
  grep count in the issue, because most `dyn Trait` raw lines appear
  inside comments or in setup/builder API surface that Charon either
  skips or that does not require lowering.

The high `Drop` and `UnwindResume` counts confirm that handling Rust's
drop / unwind semantics is a real concern for the lowering driver, not
something we can wave away.

### 6. Open questions / next steps

These are intentionally **not** answered in Step 0:

- **Charon version pinning.** The releases are all `prerelease: true`
  nightlies, so "pin a stable release" (issue §Step 2) means picking a
  specific nightly tag and committing to it. Decide which date to pin
  in Step 2 — `nightly-2026.05.24` is a reasonable starting point.
- **`dyn Trait` classification.** We have a low type-level count (7),
  but issue Step 1 still requires walking the hot-path call sites
  (especially `DictStrategy` and friends) to classify pre-refactor.
  That belongs in its own task.
- ~~**MIR → `FunctionGraph` prototype** (issue Step 0, paragraph 2).~~
  Landed under [`prototype/`](prototype/) — covers all four corpus
  functions and produces the canonical text snapshots under
  `prototype/expected/`. Comparison against the *real* AST front-end
  output is documented in `prototype/README.md` as a per-construct
  delta table rather than a direct text diff (the AST front-end is
  built for interpreter code, not arbitrary Rust like the corpus, so
  a head-to-head graph diff would be misleading at this scale).
- **`charon-lib` dependency from stable Rust.** Issue Step 2 wants the
  downstream consumer to use `serde`/`charon-lib`. Verify that
  `charon-lib` builds on our stable Rust toolchain (not the pinned
  Charon nightly). The crate is published on crates.io but its
  feature set / MSRV needs confirmation before Step 2.
- **Caching strategy.** A 133 MB ULLBC per interpreter rebuild is too
  much to regenerate per developer build. Step 2 needs to decide:
  check in extracted artifacts, or cache in CI, or both?

### 7. Acceptance check against issue #97

| Step 0 acceptance criterion                                                | Status |
|----------------------------------------------------------------------------|--------|
| Pick representative corpus                                                 | done   |
| Run Charon on corpus                                                       | done   |
| Inspect `.llbc` for locals/places/constants/calls/trait resolution/terminators/discriminants/spans/unwind | done — see §2–§4 |
| Minimal MIR → `FunctionGraph` prototype for 1–2 functions                  | done — covers all 4 corpus functions; see `prototype/` |
| Comparison against AST front-end after canonical normalization             | done — `prototype/README.md` "Mapping table" + "Deltas" sections; canonical snapshots under `prototype/expected/` |
| Checked-in notes / fixtures                                                | this README + `corpus/` + `corpus.ullbc` + `inspect_llbc.py` + `prototype/` |
