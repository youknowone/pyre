# Step 0.4 — Minimal MIR → `FunctionGraph` prototype

> **DELETION CANDIDATE (2026-05-25).**  Step 3's real MIR driver
> (`majit-translate::front::mir`) supersedes this prototype.  It is
> retained only as a Step 0.4 reference for the
> ULLBC-shape-vs-FunctionGraph mapping notes below.  The text
> snapshots in `expected/` are no longer the canonical regression
> oracle — `majit-charon-reader::tests::corpus` and
> `majit-translate::tests::test_mir_frontend` consume the same
> `corpus.ullbc` directly and are the authoritative checks.  Safe
> to delete once issue #97's Step 6.F (front::ast retirement)
> lands; corpus.ullbc + corpus/ + inspect_llbc.py + step1-
> compatibility.md stay because they have downstream consumers.

A tiny Rust crate that reads `../corpus.ullbc` (Charon's basic-block IR
for the spike corpus), lowers each function into a `FunctionGraph`-**shaped**
structure, and prints a stable canonical text form for diffing.

The prototype is intentionally a **standalone crate** (`[workspace]` in
its own `Cargo.toml` so it does not join the main workspace). It does
not depend on `majit-translate`. Its purpose is to:

1. Demonstrate that the ULLBC → CFG mapping is mechanical and small.
2. Surface every place where MIR's shape differs from
   `majit-translate::model::FunctionGraph`, so Step 3's real driver
   can plan for those gaps explicitly.
3. Produce stable text snapshots (`expected/*.txt`) that catch
   schema regressions when we re-extract the corpus with a newer
   Charon.

## Reproducing

```sh
# from this directory
./compare.sh
```

`compare.sh` regenerates each function's canonical text and diffs it
against `expected/*.txt`. Exit code 0 = all match.

## Files

```
prototype/
├── Cargo.toml          # isolated workspace, depends on serde + serde_json
├── src/
│   ├── main.rs         # CLI: charon-spike-lower <file.ullbc> <fn> ...
│   ├── ullbc.rs        # minimal serde decode of the ULLBC subset used
│   ├── graph.rs        # FunctionGraph-shaped target IR + canonical printer
│   └── lower.rs        # ULLBC → graph translation
├── expected/           # checked-in canonical snapshots (diff target)
└── compare.sh          # regenerate + diff (CI hook)
```

## Mapping table

| ULLBC concept                                | Prototype graph concept                                       | Notes vs real `FunctionGraph` |
|----------------------------------------------|---------------------------------------------------------------|--------------------------------|
| `arg_count` first locals after slot 0        | `FunctionGraph.args: Vec<String>`                              | Real graph uses `Variable` identity, not strings |
| `locals[i]` (`name`, `index`, `ty`)          | `%name_index` or `%t<index>` text label                        | Real graph holds a `Variable` per slot |
| `basic_blocks[i]`                            | `Block { id: i, inputargs: [], operations, exit }`             | **Real `Block.inputargs` are phi inputs; MIR has none** — see §Deltas |
| `Statement::StorageLive` / `StorageDead`     | dropped                                                        | No analog in `FunctionGraph` |
| `Statement::Assign(place, rvalue)`           | `SpaceOperation { result, op, args }`                          | Real `OpKind` is a typed enum tree |
| `Statement::Assert {...}`                    | dropped (overflow assert surfaces as terminator-level Assert)  | — |
| `Terminator::Return`                         | `ExitKind::Return(return_local)`                               | Real graph routes through `returnblock` sentinel via `Link` |
| `Terminator::UnwindResume` / `Abort(...)`    | `ExitKind::Resume`                                             | Real graph would route through `exceptblock` sentinel |
| `Terminator::Goto { target }`                | `ExitKind::Goto(Link { target, args: [], exitcase: None })`    | — |
| `Terminator::Switch.If(then, else)`          | `ExitKind::Switch { cases: [(true, then), (false, else)] }`    | Real graph: `ExitSwitch::Value(v)` + `Link.exitcase=Bool(t/f)` |
| `Terminator::Switch.SwitchInt(ty, [(v,bb)], dft)` | `ExitKind::Switch { cases: [(v_label, bb), ..., (None, dft)] }` | Real graph: `Link.exitcase=Const(v)` |
| `Terminator::Call { call, target, on_unwind }`| `ExitKind::Call { callee, args, dest, on_success, on_unwind }` | Real graph emits the call as a `SpaceOperation` inside the block and the unwind edge as a `Link` with `last_exception`/`last_exc_value` set on `exceptblock`-bound paths |
| `Terminator::Assert { assert, target, on_unwind }` | `ExitKind::Assert { cond, on_success, on_unwind }`        | Real graph would skip overflow asserts entirely in interpreter hot paths |
| `Terminator::Drop { target, on_unwind, ... }`| `ExitKind::Call { callee: "<drop>", ... }`                     | Real graph does not represent Rust drops at all; Step 3 driver needs an explicit policy |

## Deltas worth calling out

These are the structural deviations that have to be designed for in
Step 3, not bugs in the prototype.

### 1. **Per-block phi-inputs vs. function-wide locals**

`FunctionGraph::Block` carries `inputargs: Vec<Variable>` — phi inputs
that incoming `Link`s pass values into. MIR has no phi nodes; every
basic block reads from the same flat `locals[]` slot table.

The Step 3 driver needs an SSA-conversion pass (or a phi-introduction
pass over MIR) to populate `Block.inputargs`. The prototype short-cuts
this by leaving `inputargs = []` and letting bare local reads encode
data-flow via slot identity.

The AST front-end currently uses a different mechanism for this — its
recursive walker carries a per-scope binding map (`local_value_ids`),
and `lazy_install_local_at_current_block_var` retroactively adds
`Link.args` entries when a successor block reads a name first
established in a predecessor (see issue #97 §"Problems with the
current AST-based approach"). MIR provides the predecessor information
explicitly, which is why issue #97 expects this whole shim to retire.

### 2. **`returnblock` / `exceptblock` sentinels are absent in MIR**

`FunctionGraph::new` constructs `returnblock` (BlockId 1) and
`exceptblock` (BlockId 2) up front; every block returning a value
holds a `Link([value], returnblock)` in its `exits`. MIR has
per-block `Return` and `UnwindResume` terminators with no sentinel.

Step 3 has two options: (a) preserve the sentinel block model and
synthesise `returnblock`/`exceptblock` at lowering time; (b) retire
the sentinels and represent return/resume as first-class block
terminators. The latter is closer to MIR but a bigger change to the
downstream pipeline. The prototype follows (b) for simplicity.

### 3. **Drop terminators**

Rust drops show up as `Terminator::Drop` with `target` (success) and
`on_unwind` edges. The 22-block `desugar_mix` has eight `Drop`
terminators just to release the `slice::Iter` between iterations.
The AST front-end has no analog because `syn` AST hides drops.

The Step 3 driver needs an explicit policy: emit drops as JIT-visible
ops (slow but correct for interpreter helpers that close files or
mutexes), or strip them in the lowering pass when the type is known
to be POD (which we can read from the deduplicated type table).

### 4. **Overflow asserts**

`a + b` in debug mode lowers to
`Assign(t, BinaryOp("AddChecked", a, b))` + `Assert(!t.1)`.
`straight_line_add` has three such asserts; the JIT does not want
them. Step 3 should treat `AddChecked` as `Add Wrap` and strip the
paired `Assert` when the surrounding bb does nothing else with `t.1`.
Without this filter the lowered graph is ~3× the size of what the AST
front-end produces today for the same input.

### 5. **Constants**

The prototype prints `const(kind<…>)` for every constant — it does
not decode the `ConstantExpr` value tree because the variants are
many (literals, statics, function pointers, ZSTs, aggregates) and
only the literal forms are needed for diffing the corpus. The Step 3
driver must implement a full decoder mapping each `ConstantExpr`
variant to `model::ConstValue`.

### 6. **`Place` projections**

Projections (`Field`, `Deref`, `Index`, `Subslice`, `Downcast`, …)
are stacked on `Place.kind = Projection(inner, elem)`. The prototype
renders them as a dotted suffix (`%v.Tuple2_0.*.Adt1_0`). The real
`FunctionGraph` would emit each projection step as a separate
`SpaceOperation` (`getfield`, `getarrayitem`, etc.) — that lowering
expansion lives in Step 3, not here.

## Run on a single function

```sh
cargo run --quiet -- ../corpus.ullbc straight_line_add
cargo run --quiet -- ../corpus.ullbc desugar_mix
```

Pass any local function name from the corpus (`local_fn` matches
`ends_with("::name")`).
