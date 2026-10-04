# #346 cluster A: `PyError` is the raised value

Status: design, 2026-09-28. Decided: the RPython-level raised value becomes `PyError`, the
`OperationError` analog. This document gives the shape, the blast radius and the landing order.

## 1. Measurement

Release prepass census, `PYRE_RTYPER_VERBOSE=1 cargo build --release -p pyre-jit-trace`, fixed LLBC:

| main | Charon | phase A | phase B | Skip |
|---|---|---:|---:|---:|
| `33a06674e0c` | 2026.05.29 | 1910 | 0 | 1906 |
| `b899e2c917e` | 2026.09.26 | 1895 | 0 | 1891 |

- **Phase B:** the `PtrRepr → RootClassRepr` phase-B family that the #346 body counts at 309 no longer
  exists. Phase B is 0.
- **Phase A:** the largest exception-domain family is `UnionError … no common base class:
  pyobject::PyObject ∪ Exception`, with 181 subjects on the new pin. A smaller one is
  `Exception ∪ error::PyError`, with 5. Every one of these fails inside two graphs:
  - `pyre_object::interp_exceptions::w_exception_get_kind`: 308 hits;
  - `pyre_interpreter::error::<Impl>::from_exc_object`: 52 hits.

## 2. Root cause

Today the trace-level exception value is the app-level `W_BaseException` object, not an RPython
exception instance:

- **Raise side.** The front lowers `return Err(e)` to `raise (type(x), x)`, where
  `x = pyerror_to_exc_object(e)`. The helpers involved are `front/result_exc.rs`
  `materialize_error_to_exc_object` and `set_raise_from_instance`.
- **Catch side.** A `match` consumer of a `Result` gets `catch_and_rewrap`. That rebuilds
  `Err(PyError::from_exc_object(last_exc_value))`, and drain-match fusion calls
  `exception_object_matches_stop_iteration(last_exc_value)`.
- **Annotation of the caught value.** The annotator types every caught value as
  `SomeInstance(Exception)`. `flowin` intersects the exit case, which is the builtin `Exception`,
  and `follow_raise_link` then binds `last_exc_value`.
- **Where it breaks.** `from_exc_object` and `w_exception_get_kind` take a `PyObjectRef`. Their other
  callers pass `SomeInstance(PyObject)`. `PyObject` and `Exception` have no common base, so the
  merge fails.

## 3. Upstream shape

The upstream counterparts, all in `pypy/interpreter/`:

- **`error.py` `OperationError(Exception)`** carries `w_type`, `_w_value` and
  `_application_traceback`. `_w_value` is computed lazily. `oefmt` builds an `OpErrFmt*` subclass that
  formats its message only on demand. `normalize_exception` is `@jit.unroll_safe`.
- **`pyopcode.py` `handle_bytecode`** catches `OperationError`. `handle_operation_error` pushes
  `operr.normalize_exception(space)`, the W object, onto the value stack. `RAISE_VARARGS` wraps the
  app-level value with `OperationError(w_type, w_value)`.

On the RPython side:

- **`flowcontext.py` `exc_from_raise`:** a raise is the pair `(op.type(inst), inst)`.
- **`exceptiondata.py`:** `r_exception_value = getinstancerepr(rtyper, None)`, so the exception
  value's lltype is `OBJECTPTR`.
- **`pyjitpl.py`:** `last_exc_value` is an `OBJECTPTR`, and `handle_possible_exception` guards
  `GUARD_EXCEPTION` on `last_exc_value.typeptr`. That is the vtable of the `OperationError`
  *subclass*, not the app-level `w_type`. `opimpl_raise` does the same.
- **`blackhole.py`:** `exception_last_value` is an `OBJECTPTR`. `bhimpl_last_exc_value` returns it
  as a GCREF.

So in PyPy the value that travels through `raise`, `last_exc_value`, the JIT exception cells and
the trace is the `OperationError` instance. The W object is a field of it.

## 4. Target shape in pyre

| | today | target |
|---|---|---|
| RPython raised value | `W_BaseException` ref (via `to_exc_object`) | `PyError` GC instance |
| `PyError` storage | ~56-byte Rust value, `derive(Clone)` deep copy | one-word GC ref (handle), aliasing on clone |
| class word at offset 0 | `ob_type` = per-`ExcKind` `PyType` | a class vtable for `PyError` (subclass of `Exception`) |
| app-level class | the W object's `ob_type` | a field of `PyError` (`w_type` / `kind`) |
| `GUARD_EXCEPTION` const | per-`ExcKind` `PyType` | `PyError` class vtable |
| `last_exc_value`, `BH_LAST_EXC_VALUE`, backend `JIT_EXC_VALUE` | W object | `PyError` ref |
| W object | the value itself | `PyError.exc_object`, lazily materialised (`get_w_value`) |
| annotator classdef | `PyError` is a root class (no base) | `PyError` has base `HOST_ENV Exception` |
| front `ErrorCarrierSpec` | `to_exc_object` / `from_exc_object` set | both `None` (identity path, already supported) |

Constraints that force the runtime part:

- **The class word.** Every backend's `jit_exc_raise`, `bh_classof`, `bh_issubclass`,
  `raise_if_successful` and `walker_record_guard_exception` read offset 0 of the value as a class
  word. The value is also rooted and forwarded as a movable GC object. The raised value therefore
  has to be a GC object whose first word is its class, and that is what an RPython `OperationError`
  is. A by-value Rust struct cannot be the raised value.
- **The `'r'` kind.** The codewriter's `raise/r`, `last_exc_value/>r` and exceptblock slot 1 are
  `'r'`, and a `PyError` handle stays one word, so none of that changes.

## 5. Design points

### 5.1 The `PyError` object

A GC struct allocated through the ordinary GC hook. Its fields keep today's meaning:

| field | meaning |
|---|---|
| class word | `PYERROR_TYPE`, a static class object with its own subclass range, registered below the `Exception` node (`majit_gc::subclass_range`) |
| `kind` | the app-level class |
| `message` | lazily formatted text; the `OpErrFmt` role |
| `exc_object` | `_w_value` |
| `attach_tb`, `context_recorded`, `reraise_lasti` | as today |
| `w_name_context`, `w_obj_context` | as today |

The traceback stays on the W object, where pyre keeps it today. That is an existing adaptation and
it is out of scope here.

### 5.2 Rust API

`pub struct PyError(GcRef)` keeps its methods. `Clone` becomes aliasing, which is `OperationError`
semantics. Every `clone()`-then-mutate site has to be audited, because a mutation on a clone becomes
visible through the original.

### 5.3 Rooting

`PyError` already holds a movable GC ref, `exc_object`. The code roots it through
`pin_exc_object`, `publish_exc_object` and `reload_exc_object` around collecting calls. After the
change every live `PyError` is such a ref, including the ones whose W object has not been
materialised yet, so those brackets now cover the handle itself.

Two ways to do that are open (**decision needed**):

- **(a) Brackets at collecting call sites,** the existing gc-decouple discipline.
- **(b) An RAII pin** (`majit_gc::pin` / `unpin`) held by the handle.

Option (b) is simpler to apply, but it is a pyre-only root owner with no upstream counterpart,
because PyPy's shadowstack roots the value automatically. Option (a) is the orthodox one.

### 5.4 JIT runtime ABI

- **Publish sites.** Every `jit_publish_exception(err.to_exc_object())` and
  `publish_residual_call_exception` site publishes the handle instead.
- **Read sites.** Every `from_exc_object(value)` site adopts the handle. There are about 180 publish
  sites and 95 adopt sites; see §6.
- **Readers of the app-level class.** Readers such as `w_exception_get_kind(exc)` in
  `pyre-jit-trace` specialisations read `PyError.kind` instead. That is a traced getfield. Readers
  that need the W object go through `get_w_value`.
- **The interpreter.** `handle_exception_with_context` keeps pushing the W object
  (`normalize_exception`), which is `pyopcode.py handle_operation_error`.

### 5.5 Translator

- **Base class.** `PyError`'s ClassDef gets `Exception` as base. `bookkeeper.rs`
  `intern_class_by_qualname_with_bases` exists but is unused for this. It must run before the
  session prologue pre-mints `PyError`.
- **Carrier.** In `prepass.rs` the error carrier gets `to_exc_object: None` and
  `from_exc_object: None`. The identity path already exists (`lib.rs ErrorCarrierSpec`).
- **Front passes.**
  - `catch_and_rewrap` forwards `last_exc_value` as the `Err` payload.
  - `err_payload_is_dead` loses its reason to exist.
  - Drain-match fusion tests the `PyError`: `exception_object_matches_stop_iteration` takes the
    handle.
  - `fuse_kind_ctor_raise` and its `pyerror_*_to_exc_object` table are deleted.
  - `result_map_err`'s round-trip bypass is deleted.
- **Registry and rclass.**
  - The `call_registry.rs` materialiser entries (`is_exception_object_materializer`) are deleted.
  - The `rclass.rs` `to_exc_object` method arm and `unaryop.rs` `pyerror_method_to_exc_object` stay
    only as long as interpreter code still calls `to_exc_object` explicitly.

## 6. Blast radius

Counts from `rg` over the tree (approximate):

| area | sites |
|---|---:|
| backends (dynasm, cranelift, wasm, majit-backend) | ~150 |
| majit-metainterp | ~340 |
| pyre-jit | ~170 |
| pyre-jit-trace | ~365 |
| pyre-interpreter JIT boundary | ~90 |
| interpreter `to_exc_object` / `from_exc_object` | ~180 / ~95 |
| majit-translate | ~25 |

Most backend and metainterp sites are type-agnostic. They move an `i64`/`GcRef` and read the class
word, so they need no edit once the class word is right. The real edits are:

- the publish and adopt sites;
- the `pyre-jit-trace` specialisations that read `ExcKind` off the value;
- the walkers that root the value (`walk_raw_exception_roots` walks a `PyError` first, then its
  W object);
- `memory_error_singleton_ref`, which becomes a prebuilt `PyError`.

## 6a. Runtime facts P1 builds on

- **Class word.** `PyType` is laid out as the rclass `OBJECT_VTABLE`: `subclassrange_min` /
  `subclassrange_max` sit at its start (`pyobject.rs`). `bh_issubclass` looks both classes up with
  `majit_gc::subclass_range`. Exception kinds are registered in `build_gc` (`pyre-jit/src/eval.rs`)
  from the `exc_hierarchy` table, `SUBCLASS_RANGE_HIERARCHY` and `all_subclass_range_aliases`.
  - So the `PyError` class word is a `PyType`-shaped static, registered the same way.
  - An RPython-level `Exception` root node is its parent. That node is not the app-level
    `BaseException` range.
- **Traced allocation.** `model::fuse_boxing_alloc` turns `lltype::malloc_typed(S { ob_header: PyObject
  { ob_type: &S_TYPE, .. }, .. })` into `NewWithVtable` plus setfields; `W_ComplexObject` is the
  model to copy. Allocating the `PyError` object this way keeps it virtualizable in traces, as
  `OpErrFmt` is in PyPy. Anything allocated through `try_gc_alloc*` is residual.
  - The steps to copy: `GcType`, `build_gc` registration, the subclass-range tables,
    `jit_static_pytype_addrs`, and a `descr.rs` group.
- **Message storage.** A GC object cannot own a Rust `Wtf8Buf`, because nothing would drop it. The
  message text moves to an `rstr` low-level string (`lowlevel_string.rs`, `STR`/`UNICODE`). The
  per-format subclasses (P1b) instead hold the argument values and format on demand.
- **Rooting inventory.** `scripts/check-gc-root-brackets.py`, built on the `gc-root-reachability`
  example, reports GC refs that are live across a collecting call without a bracket.
  - Registering the `PyError` handle as a GC pointer type makes that tool list every `PyError` site
    that needs a bracket.
  - The two invariants the tool holds at zero must stay at zero, and its backlog count must not rise.
- **Latent defect found on the way.** The codewriter emits an exception-class constant
  (`goto_if_exception_mismatch` `llexitcase`, the `raise` of `overflow_error_instance`) as
  `HostObject::identity_id()`. That is a build-process pointer, and no load-time patch rewrites it.
  The rtyped `llexitcase` (`RootClassRepr.convert_const`) never reaches the codewriter either.
  - Fix this before P2. After P2, typed catches compare against RPython class vtables, so these
    constants must be real runtime addresses.
  - Upstream reference: `flatten.py insert_exits` emits `link.llexitcase`, the rtyped vtable.

## 7. Landing order

Each step is a PR that is green on `cargo test --all --no-default-features --features
dynasm,pyre-module`, a bare `pyre/check.py` (all three backends) and
`scripts/check-rtyper-skip-subjects.py`.

0. **P0 — exception-class constants.** Carry the rtyped `llexitcase` to the codewriter, and map
   RPython exception class vtables to runtime class addresses (the latent defect in §6a).
1. **P1 — `PyError` becomes a GC object, behind the same boundary.**
   - Make `PyError` a handle to the GC struct, with rooting per §5.3.
   - Audit `Clone` sites.
   - The JIT boundary still converts with `to_exc_object` / `from_exc_object`, so trace shapes and
     jitstats do not move.
   - Needs an LLBC re-extract and a census, because interpreter source shapes change.
2. **P2 — the raised value switches, runtime and translator together.** A mixed state (jitcode
   raising W objects while the runtime cells expect a `PyError`) is a miscompile, so this cannot
   be split.
   - Publish and adopt the handle.
   - Guard on the `PyError` class.
   - Read `kind` through a field.
   - Carrier identity; base class `Exception`.
   - Delete the front round-trip code.
   - Before re-recording any jitstats, compare trace shapes (`GUARD_EXCEPTION` const, getfield of
     `kind`) with the PyPy oracle on the exception fixtures (`exception_*`, `pickle_terminal_*`).
3. **P3 — cleanup.** Delete the `to_exc_object` annotator/rtyper special arms that no longer have
   callers, and re-census.

Done when the `PyObject ∪ Exception` and `Exception ∪ PyError` families are gone from the census,
with no new Skip names and no compensating union arm.

## 8. Decisions (2026-09-28)

1. **Rooting: brackets (a).** A live `PyError` handle is rooted around collecting calls with the
   gc-decouple `RootScope` discipline, the way `exc_object` is today. No RAII pin.
2. **Classes follow `OpErrFmt`.**
   - `PyError` is the base class. It corresponds to `OperationError` and carries `kind`,
     `exc_object`, the flags and the contexts.
   - A formatted error is an `OpErrFmt`-style subclass per format: `get_operr_class(valuefmt)`. It
     stores the format's literal pieces (`xstrings`) and its arguments (`x0..xN`), and builds the
     message only in `_compute_value`.
   - An error without format arguments is the `OpErrFmtNoArgs` subclass.
   - `GUARD_EXCEPTION` is then keyed on the subclass vtable, as in `pyjitpl.py`
     `handle_possible_exception`, and the exception class check becomes `bh_issubclass` against
     `PyError`.
   - P1 introduces the base class and `OpErrFmtNoArgs`. The per-format subclasses are a P1b step
     before P2, so that P2 changes only which value is raised.
