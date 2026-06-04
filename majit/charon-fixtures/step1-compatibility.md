# Step 1 — Compatibility classification

Issue #97 Step 1 says:

> Classify every Charon-blocking language feature by whether it is
> actually consumed by the JIT lowering pipeline. Refactor only the
> hot-path blockers first.

The Step 0 extraction (see `README.md`) already proved that
**Charon successfully extracts `pyre-interpreter` and `pyre-object`
end-to-end** (only `thread_local!` accessor stubs fail). That changes
the priority of "blockers" significantly compared to the issue's
opening text — Charon does not blow up on `dyn Trait` or
`impl Trait` returns as the issue feared. So this document classifies
each candidate against **two** criteria:

1. **Extraction compatibility** — does Charon represent it usefully in
   `.ullbc`?
2. **Lowering compatibility** — can the Step 3 driver lower the
   resulting MIR shape into a `FunctionGraph` that the JIT can run?

A site that Charon extracts but the lowering driver cannot consume is
still a refactor candidate — but it is not a *Charon* blocker.

## 1. `dyn Trait` survey

### 1.1 Source counts (JIT-consumed crates only)

`pyre-interpreter`, `pyre-object`, `pyre-module` — the crates whose
code becomes JIT bytecode. `pyre-jit*` is JIT machinery, not lowered,
so it is excluded from the hot-path count.

| Trait                    | Sites | Impls         | Verdict                                                  |
|--------------------------|------:|---------------|-----------------------------------------------------------|
| `AsyncActionOps`         | 14    | 3             | Registry of pluggable signal/GC actions; truly polymorphic |
| `ActionFlagOps`          | 9     | **1** (`ActionFlag`) | Trivially monomorphizable                            |
| `FnMut`                  | 6     | (closures)    | GC tracing visitors inside `unsafe` blocks; cold path     |
| `PeriodicAsyncActionOps` | 4     | 1 (`PeriodicAsyncAction`) | Trivially monomorphizable                       |
| `DictStrategy`           | 3     | **8**         | Issue's named hot-path blocker; per-method virtual dispatch on every dict op |
| `DictStorageErased`      | 1     | (per-strategy) | Companion to DictStrategy — type-erased storage         |
| `Write`                  | 1     | (impl fmt::Display) | Cold formatter                                      |
| `Any`                    | 1     | (std)         | Cold downcast                                             |

**Total: 39 `dyn Trait` occurrences across 8 distinct trait identities.**

### 1.2 Extraction reality (from `pyre-interpreter.ullbc`)

Charon distinguishes three kinds of function call in extracted bodies:

| `Call.func.Regular.kind` | Count    | Meaning                                             |
|--------------------------|----------|------------------------------------------------------|
| `Fun.Regular(<id>)`      | 25,502   | Regular monomorphized function call                  |
| `Trait`                  | 382      | Trait-bound generic call (static, with trait clause) |
| `Dynamic(<operand>)`     | **36**   | **`dyn Trait` virtual call** (fat-pointer dispatch)  |

The Dynamic calls are the ones the issue cares about. They are concentrated
in just 28 caller functions, top of which:

```
  7  pyre_interpreter::function::funccall_valuestack
  4  pyre_interpreter::executioncontext::fire
  2  pyre_interpreter::executioncontext::ActionFlagOps::action_dispatcher
  2  pyre_interpreter::executioncontext::action_dispatcher
  2  pyre_interpreter::executioncontext::new
  …
```

**Observation:** zero of the 36 Dynamic calls are `DictStrategy::*` —
the 8-impl strategy hierarchy that the issue called out as the primary
hot-path target. The dict dispatch goes through generic helpers and
trait-bound generic calls (`Trait`-kind, not `Dynamic`-kind). The
actual `&dyn DictStrategy` indirection is constrained to:

- struct field storage (`W_DictMultiObject.dstrategy: &'static dyn …`)
  — no call at all, just a fat pointer in memory.
- the `w_dict_get_strategy()` `#[inline]` accessor whose body Charon
  may have inlined away at extraction time.

This means the issue's framing — "`DictStrategy` must be enum-ified
before Charon can proceed" — is **not strictly true** for extraction.
Whether it is true for **lowering** depends on what Step 3's driver
does with `Dynamic` call terminators. See §1.4.

### 1.3 Site-by-site classification

#### `dyn DictStrategy` — 3 source sites, 8 impls

| Site                                                  | Role                                        |
|-------------------------------------------------------|---------------------------------------------|
| `dictmultiobject.rs:84` `fn get_strategy(&self) -> &dyn …` | trait method declaration on `W_DictMultiObject` |
| `dictmultiobject.rs:93` `fn set_strategy(&mut self, &'static dyn …)` | trait method declaration                 |
| `dictmultiobject.rs:144` `struct W_DictObject { dstrategy: &'static dyn … }` | struct field                         |

Impls (all in `pyre-object`):
`IdentityDictStrategy` / `KwargsDictStrategy` /
`EmptyKwargsDictStrategy` / `EmptyDictStrategy` /
`ObjectDictStrategy` / `BytesDictStrategy` /
`UnicodeDictStrategy` / `IntDictStrategy`. Eight in total.

**Refactor verdict:** **deferred**. The issue's framing assumed a
specific Charon failure mode that did not materialize. Until Step 3
proves the lowering driver cannot deal with `Dynamic` calls on the
dict path, the enum refactor is premature optimization that risks the
parity-sensitive concerns the issue itself catalogues:

> `DictStrategy` conversion must preserve strategy identity,
> module/celldict behavior, GC tracing, cache invalidation, and all
> observable dict semantics.

Re-evaluate after Step 3 lands. If the lowering driver can devirtualize
`Dynamic` calls via type-flow analysis (the natural PyPy-style approach,
treating the fat pointer's type tag as a guard), the refactor is
unnecessary.

#### `dyn ActionFlagOps` — 9 sites, **1** impl

Only `ActionFlag` implements `ActionFlagOps`. Every `dyn ActionFlagOps`
site is morally a `&mut ActionFlag` with extra ceremony.

**Refactor verdict:** **easy retire** when convenient — replace with
the concrete `ActionFlag` type and drop the trait. This is mechanical
and risk-free, but it is also not gating anything Charon-related, so
it can wait. Treat as a Step 1.4 polish task, not a Charon blocker.

#### `dyn AsyncActionOps` (14) + `dyn PeriodicAsyncActionOps` (4)

`AsyncActionOps` has **3** impls (`AsyncAction`, `PeriodicAsyncAction`,
`UserDelAction`). The registry shape is
`Vec<*mut dyn AsyncActionOps>` (`executioncontext.rs:1542`) — pluggable,
designed for runtime extension. `PeriodicAsyncActionOps` has 1 impl
(`PeriodicAsyncAction`).

**Refactor verdict:** **keep as `dyn` for now**. The registry shape is
*the point* — it lets the GC, signal handler, and user-del action
plug in uniformly. Converting to an enum would force every future
async action to land in the enum (i.e., in `pyre-interpreter`), which
breaks the layering. The 36 Dynamic calls in the extracted ULLBC are
not a problem to extract; whether the JIT can fold them through is a
Step 3 question. If it cannot, the answer is probably to leave them
as opaque indirect calls in the JITed code (they fire ≤ once per N
opcodes — the dispatch cost is amortized).

#### `dyn FnMut` — 6 sites

All 6 sites are GC tracing visitors:

```
pyre-object/src/dictstrategy.rs:234   walk_gc_refs(... visitor: &mut dyn FnMut(*mut PyObjectRef))
pyre-object/src/dictstrategy.rs:935   walk_gc_refs(... visitor: &mut dyn FnMut(*mut PyObjectRef))
pyre-object/src/dictstrategy.rs:1265  walk_gc_refs(... visitor: &mut dyn FnMut(*mut PyObjectRef))
pyre-object/src/kwargsdict.rs:238     walk_gc_refs(... visitor: &mut dyn FnMut(*mut PyObjectRef))
pyre-interpreter/src/eval.rs:166      visitor: &mut dyn FnMut(&mut majit_ir::GcRef)
pyre-interpreter/src/eval.rs:187      walk_pyframe_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef))
```

These are called from `majit-gc` during stop-the-world tracing, not
from any opcode handler the JIT lowers. They have to be `dyn` because
each invocation walks heterogeneous objects with one callback.

**Refactor verdict:** **leave as `dyn`**. Cold path, GC-internal,
correctly modelled today.

#### `dyn Write` / `dyn Any` — 2 cold sites

Single occurrences in formatter / downcast code. **Leave as `dyn`**.

### 1.4 What the Step 3 driver actually has to handle

When the Step 3 driver encounters
`Call.func.Regular.kind.Dynamic(<operand>)` it has three options:

1. **Opaque indirect call.** Emit a `direct_call`-shaped operation
   whose callee is the runtime fat-pointer value, drop the JIT's
   ability to inline through it, and rely on the called code being
   short enough that the indirect-call overhead is acceptable.
2. **Type-flow specialization.** If the type system can prove the
   concrete type behind the fat pointer at a given call site (e.g. by
   threading the strategy singleton through a per-dict guard), rewrite
   the Dynamic call as a direct call to the monomorphized method —
   the PyPy-style approach.
3. **Source refactor.** What the issue originally proposed: turn the
   trait hierarchy into an enum, then the call becomes
   `match strategy { Foo => Foo::method(...), Bar => Bar::method(...) }`
   which is plain `Fun.Regular` everywhere.

Step 3 should attempt (1) first (it is unconditionally correct and
small), then (2) for hot paths if the JIT shows the indirect call
dominating profile, then (3) only as a last resort because it disrupts
the layering described above.

## 2. `impl Trait` return-position survey

The issue listed 9 sites of `impl Trait` in return position. They split
clearly:

| Site                                                              | Role                                  | Lowered? |
|-------------------------------------------------------------------|---------------------------------------|---------|
| `pyre-object/celldict.rs:371` `iter_values_mut() -> impl Iterator` | dict iteration, called from interp    | yes — hot |
| `pyre-object/celldict.rs:739`                                     | dict iteration                        | yes — hot |
| `pyre-object/celldict.rs:755`                                     | dict iteration                        | yes — hot |
| `pyre-interpreter/executioncontext.rs:445` `entries() -> impl Iterator` | dict iteration                  | yes — hot |
| `pyre-interpreter/executioncontext.rs:590` `keys() -> impl Iterator`    | dict iteration                  | yes — hot |
| `pyre-jit-trace/jitcode_runtime.rs:676`                           | JIT-runtime                           | no       |
| `pyre-jit/jit/codewriter.rs:10687` `fresh_variable_factory()`     | JIT codewriter setup                  | no       |
| `pyre-jit/jit/flatten.rs:5701` `identity_register_mapper()`       | test helper                           | no       |
| `pyre-jit/jit/flatten.rs:5717` `test_constant_lowering()`         | test helper                           | no       |

**Extraction reality:** all 9 functions were extracted successfully
by Charon in the Step 0 run. `impl Iterator` returns become anonymous
opaque types in MIR (one per call site), which Charon serializes as
ordinary `Deduplicated` type references.

**Refactor verdict:** **no refactor needed**. The five `iter_*`
returns are unconditionally fine — they desugar to ordinary structs
in MIR. The two JIT test helpers and two JIT-machinery helpers are
not lowered anyway. None of these are Charon blockers and none of
them are Step 3 driver concerns.

## 3. RPITIT / GAT / trait alias

Issue #97 §"Charon-supported feature set" listed these as zero hits.
The Step 0 ULLBC extraction confirms: **no RPITIT, no GATs, no trait
aliases** in any of the JIT-consumed crates. Nothing to do.

## 4. The `thread_local!` accessor gap

The only real extraction failures in the Step 0 run were 28 (pyre-object)
+ 70 (pyre-interpreter) `*::{const}::call` accessor closures generated
by the `thread_local!` macro:

```
warning: charon does not support thread local references
 --> /rustc/library/std/src/sys/thread_local/native/mod.rs:68:25
```

These are **not** in pyre source. They are inside std's `thread_local!`
expansion. Charon's error is per-statement, surfacing as
`body: { "Error": ... }` on the accessor closure only — the surrounding
pyre function (which called `MY_TLS.with(|tls| ...)`) extracts cleanly.

**Refactor verdict:** the Step 3 driver must treat `thread_local!`
accessor calls as opaque ops. No refactor in pyre source. Track upstream
[charon#1??] (TLS support) if/when it becomes a blocker.

## 5. Summary

| Concern                              | Status                                           |
|--------------------------------------|--------------------------------------------------|
| `dyn DictStrategy` refactor (Step 1) | **deferred** — not actually a Charon blocker; Step 3 driver decision |
| `dyn ActionFlagOps` (single impl)    | low-risk mechanical retire whenever convenient   |
| `dyn AsyncActionOps` (3 impls, registry) | **keep as `dyn`**; opaque call in JIT          |
| `dyn FnMut` (GC visitors)            | **keep as `dyn`**; cold path                     |
| `impl Trait` returns (9 sites)       | **no refactor**; Charon extracts them fine       |
| RPITIT / GAT / trait alias            | none in scope                                     |
| `thread_local!` accessor opacity     | Step 3 driver must accept opaque TLS ops          |

**Net result for the Charon migration:** no source-level refactor is
strictly required to *unblock* the Charon pipeline. The Step 3 driver
should be designed against the **as-extracted** shape of the ULLBC,
with `Dynamic` calls treated as first-class indirect-call terminators.
The strategy/action enum-ification refactors discussed in the issue
remain on the table but are now **post-cutover polish**, not gating
work.

### Next steps (Step 1 closed; Step 2 unlocked)

- [x] Step 1.1 — `dyn Trait` survey (this doc)
- [x] Step 1.2 — `impl Trait` return survey (this doc)
- [ ] ~~Step 1.3 — `DictStrategy` refactor~~ → **deferred** per §1.4
- [ ] Step 2 — Charon integration scaffolding
