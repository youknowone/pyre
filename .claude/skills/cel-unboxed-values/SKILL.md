---
name: cel-unboxed-values
description: The design of record for task #52 — converging cel-jit onto ONE class-based CEL bytecode VM with an Arc-free, W_Root-shaped value universe traced by majit-translate (front-end B). Use this whenever working on cel's value representation, its bytecode VM, its JIT portal, or the majit changes those need (#55 virtualizable frame, #82 list boxing, #83 the missing portal-entry warm-up door, #84 neutral tracer, #85 trait families). Read it BEFORE proposing a value-representation change, and follow its phase order — the shapes in §2 look right and do not lower.
---

# Converge cel-jit onto one class-based CEL bytecode VM

Design of record for **task #52**. Produced 2026-08-06/07 by a 19-agent design
workflow (5 parallel subsystem readers → 3 independent designs → 3-lens
adversarial judging → synthesis → completeness critic), against the vendored
`rpython/`/`pypy/` oracle.

## How to read the citations

Two provenances, and they are **not** interchangeable:

* **`[verified]`** — opened in the session that wrote this file, by me.
* everything else — reported by a design agent. Line numbers drift; one known
  case: `analyze_multiple_pipeline_from_llbc_with_modules` is at
  `majit-translate/src/lib.rs:512`, cited as `:511` throughout the workflow.
  **P0.e re-verifies the twelve load-bearing ones by SYMBOL, not by line**, and
  records the drift. A citation that fails re-verification is a stop, not a
  detail.

No number in this document was produced by a build or benchmark run in the
session that wrote it. **P0 is the go/no-go for the whole plan.**

---

## 1. The decision

One interpreter: a full CEL bytecode VM as `Program::execute`, traced through
majit-translate (front-end B). The value universe becomes a `#[repr(C)]`,
header-first, **`Arc`-free** class family — the `W_Root` shape — allocated
through pyre's `lltype::malloc_typed` spelling so `fuse_boxing_alloc` mints
`NewWithVtable` and OptVirtualize can delete it.

Measured 2026-08-06 (inherited; re-checked at P0.b):

| | AST walker (`objects::Value::resolve_val`) | typed VM control (`clean_interp_seeded_f`, `bytecode.rs:1437`) |
|---|---|---|
| jitcodes | 40 | 9 (13 for `run_mainloop_f`) |
| ops | 3310 | 1873 |
| residual_call\* : inline_call\* | **577 : 73** (8:1) ⚠ measures **4.5:1** today | — |
| `guard_class` | **0** | 0 |
| dangling `vtablemethodptr` | **49** | 0 |
| ULLBC Drop / Call terminators / fns | **1248** / 648 / 100 | **4** / 222 / 37 |
| real-computation ops | — | 36% (binop 328 + arrayread 285 + arraywrite 69) |

⚠ **RE-DERIVED 2026-08-08 at cel-jit `28209d4`, against a freshly extracted
scoped `cel-portals.ullbc` (#107). The two columns now have opposite standing —
read them differently.**

| | doc, 2026-08-07 (see the §1 run) | measured, cel-jit `28209d4`, 2026-08-08 |
|---|---|---|
| typed VM `clean_interp_seeded_f` jitcodes | 9 | **9** ✅ reproduces exactly (61 insns) |
| AST walker, every cell | 40 / 3310 / 577:73 / 0 / 49 | **83 / 4349 / 656:145 / 0 / 3** — 4 of 5 cells miss ❌ |

**Both numbers are kept because the gap between them is information about when
this document was written; overwriting one destroys that.**

⭐ The scoped-artifact rule below is no longer only a warning — it is measured on
a *second* portal. The same `clean_interp_seeded_f` gives **12 jitcodes / 66
insns whole-crate** and **9 / 61 scoped**. Both artifacts recorded the same
source hash (`f51805ed070b…`) that day, so that pair has **no source variable**
in it. ⚠ It is not literally *scope-only*: `cel.ullbc` also carries
`--opaque cel::parser` and `cel-portals.ullbc` does not. Measured inert for both
portals — the scoped artefact contains zero `cel::parser` occurrences, and an
opaque declaration has no body to lower — so the comparison holds, but say "no
source variable" rather than "scope-only" unless the flags have been read.

⛔ **The walker column was blocked by a defect, not by a missing harness
(#121) — now FIXED ("majit-translate: type a borrow of a fieldless enum as Int,
like the enum itself"; `6eca81d5cef` as of 2026-08-11, branch-local, so find it
by that subject) and MEASURED.** Seeding the pipeline at
`["objects","Value","resolve_value"]` resolved and code-wrote 47 graphs, then
asserted in `codewriter/flatten.rs:1236` — *"switch exitswitch must be int
(graph `cel::objects::<Impl>::str`)"*. The cause was `front/mir.rs:4874-4877`:
the fieldless-enum discriminant fold tests the place TYPE and then consumes the
place VALUE, and through a `Deref` those disagreed, so the switch operand was
the reference. `TsOp::str` is an ordinary `fn f(&self) { match self { … } }` on
a C-like enum. The fix types a *borrow* of a fieldless enum as `Int`, like the
enum itself — `Rvalue::Ref` is already the identity, so only the declared type
disagreed with the representation being produced.

⛔⛔ **Post-fix, four of the five checkable walker cells DO NOT reproduce:**

| §1 walker cell | doc | scoped | whole-crate | |
|---|---|---|---|---|
| jitcodes | 40 | **83** | **82** | ❌ 2.1x |
| ops (less `-live-`) | 3310 | **4349** | **4347** | ❌ 1.3x |
| residual : inline | 577 : 73 | **656 : 145** | **656 : 144** | ❌ (8:1 → 4.5:1) |
| `guard_class` | 0 | **0** | **0** | ✅ |
| dangling `vtablemethodptr` | 49 | **3** | **3** | ❌ |
| `new_ops` / `new_with_vtable` | 65 | **88 / 0** | **88 / 0** | ❌ |

⚠ The one cell that reproduces **corroborates nothing**: §2 argues
`guard_class = 0` is *structural* for `dyn Val`, so it reads 0 under any tree,
scope, or source version. A cell that cannot vary is not evidence.

⚠ This is **not** the `-live-` counting trap again and not source drift. The
typed-VM column lands every structure-sensitive cell exactly with only `ops`
moving <1%; here `jitcodes` is 2.1x and `vtablemethodptr` is 49→3. Structure
counters do not drift, and no convention mismatch moves `jitcodes` while leaving
`guard_class` at 0. **This column measures a code shape the present tree does
not produce** — expected, since P2 landed 59 host-function overloads and renamed
`resolve_val` → `resolve_value`.

⛔⛔ **SCOPE IS REFUTED as the explanation, and it does NOT behave like the
typed-VM column.** The walker's whole-crate-vs-scoped delta is **1 jitcode, 2
ops, 1 inline_call** — and the jitcode moves the *opposite* way (whole-crate 82
**<** scoped 83), where the typed VM moved 16→13 and 12→9. Scope is worth ~1
jitcode here; it cannot account for 43. **The scoped-artifact rule above is
portal-specific and must not be generalised across seeds.**

⚠ **The absolutes are tree-state-conditioned; the refutation is not.** They were
measured with six other agents' uncommitted majit files in the binary, two of
them (`codewriter/call.rs`, `rtyper/rpbc.rs` = `lower_indirect_calls`) on the
measured lowering path. Method: a **delta on one frozen binary** — built once,
executable hashed, both legs run against that binary directly, hash verified
identical afterwards.

⛔ **A delta is offset-immune only if the perturbation is independent of the
treatment, and here it is not obviously so** — the in-flight edit is *in* the
pass whose input set the treatment changes. The refutation survives on
**magnitude**, not on the general principle: for a one-line predicate change to
mask a scope effect big enough to explain **43 jitcodes** while the measured
delta is **1**, its effect would have to vary with scope by roughly two orders of
magnitude more than its own total effect. That is not an interaction; it is a
different program. **Quote the scope refutation; do not put 83/82 in the table
above until they are re-derived from a clean tree.**

⚠ **The two artefacts differ in TWO extraction dimensions, not one** — checked,
because "scope-only" was inherited rather than verified. They share a source
hash (`f51805ed070b…`, both `.fingerprint` files), so there is no source
variable; but `cel.ullbc` carries `charon_flags=--opaque cel::parser` and
`cel-portals.ullbc` does not. **The flag is inert for this census**: the scoped
artefact contains **zero** `cel::parser` occurrences (the `--start-from` closure
never reaches it), and in the whole-crate artefact `cel::parser` is opaque, so it
has no bodies to lower in either leg. An opaque declaration cannot become a
jitcode. ⇒ the measured delta is attributable to scope after all — but by
measurement, not by assumption.

⭐ The pre-fix one-sided bound predicted this to the unit. With `TsOp::str`'s
body nulled the pipeline emitted **82**, and nulling can only *shrink* a
closure, so an unblocked run had to emit **≥ 82**. It emits **83** — exactly one
more, and that one is the body that had been removed.

Per-call the typed VM already beats the walker on 24 of 28 lowered cases, 1.5–14x.

⚠ **The denominator "28" is wrong** [verified 2026-08-07]. The per-call harness
builds **29** cases (12 in the initial vec + 14 from the size ladders + 3 pushed,
`cel/examples/majit_vs_cometkim_percall.rs:197-339`), while its own doc comment
at `:194` says "18 benchmark expressions" — a third number, and the one that
counts *distinct expressions* with the ladders collapsed. Every "N/28" inherited
from earlier notes (including #83's "16/28") is against a denominator that does
not exist. Re-derive before quoting.

**Why class-based and not enum-first.** Front-end B's only general
enum-variant→`New` lowering is hard-anchored to `core::result::Result`
(`front/result_exc.rs:305-321` `result_ctor_kind`, consumed at
`codewriter/jtransform.rs:3038-3076`). `CelValue::Int(x)` would arrive at the
optimizer as a `SyntheticTransparentCtor` residual — invisible to OptVirtualize,
whose entire surface is the five allocation opcodes (`virtualize.py:207-224`;
majit dispatch `optimizeopt/virtualize.rs:1988-1990`). An enum also forfeits
`known_class`, the same `InstancePtrInfo` field that carries `_is_virtual`
(`info.py:313-345`).

**Why not keep `dyn Val`.** Rust puts the vtable in the fat pointer, so there is
no field for a `getfield typeptr` to read — `guard_class = 0` on the walker is
**structural**, independent of #85. And the twelve `as_adder`/`as_comparer`/…
accessors (`cel/src/common/value.rs:11-62`) mint a *second* fat pointer per
operation; a guard on the receiver can only pin the first. Upstream has one
vtable per object and installs `__add__` into it (`pypy/interpreter/typedef.py:240-269`,
`:113-125`). RPython forbids multi-root dispatch outright
(`rpython/annotator/classdesc.py:549-550`) and shares code by *copying* methods
(`import_from_mixin`, `rpython/rlib/objectmodel.py:1104-1157`).

---

## 2. Four shapes that look right and do not lower — do not re-derive them

**(a) A function-pointer vtable struct (`CelClass { add: BinOp, … }`) does not
lower.** A call through a fn-typed struct field reaches
`(CallClass::Dynamic, CallFunc::Dynamic)` at `front/mir.rs:7938-8002`.
`dyn_indirect_target` (`mir.rs:5913-5949`) needs the terminal projection to be a
Charon-generated `::{vtable}` field — a hand-written struct is not that.
`operand_is_fn_ptr` (`mir.rs:13024-13068`) is hard-keyed to pyre's
`gateway::BuiltinCodeFn`: `inputs.len() == 1` (`:13052`), and the type strings
must contain `PyObject` / `Result<` / `PyError` (`:13064-13068`). Falling through
gives `CallTarget::FunctionPath{ segments: ["__dyn_call"] }` (`mir.rs:7990`),
documented at `mir.rs:16480-16481` as *"not a lowering, it is a placeholder: an
unregistered synthetic path that stops whatever graph reaches it"*; the
`CallKind::Ptr` bucket instead returns `LowerError::Unsupported`
(`mir.rs:7999-8002`). Both stop the graph. Even after widening,
`translator/rtyper/rpbc.rs:326-344 lower_indirect_calls` fills the candidate list
only from `call_control.builtin_wrapper_indirect_graphs()`, filtered on
`leaf.starts_with("__pyre_wrap_")` (`codewriter/call.rs:4670-4681`) — a cel slot
can never be a candidate, so `bytecode_for_address` misses and the call degrades
to residual (`pyjitpl.py:2174-2186`). And the `static CelClass{ add: … }`
initializer naming the slot functions is skipped outright (`mir.rs:904`, `:14038`),
so the slot bodies are not even in the portal's graph closure.

**(a′) The same defect applies to CEL host functions — see §6.2.** The design's
own `co_funcs: Box<[HostFn]>` table was an instance of (a). It is corrected there.

**(b) `virtualizables: []` on a stack VM forces every intermediate.**
`optimize_setarrayitem_gc` removes the store only when the *array* is virtual;
otherwise it materializes the value operand and keeps the op
(`optimizeopt/virtualize.rs:988-1023`; upstream `virtualize.py:298-308`). The
frame is built by the `&self` wrapper before portal entry, so it arrives as a
heap red inputarg and is never virtual. rsre is not a counter-example: its ctx
holds integer positions with `_immutable_fields_ = ['end']`
(`rsre_core.py:105-123`), not an operand stack of boxed objects. PyPy's actual
analogue declares `virtualizables=['frame']` with `locals_cells_stack_w[*]`
(`pypy/module/pypyjit/interp_jit.py:25-31, :67`) and `virtualizable.rst:9-24`
says it verbatim: *"virtualizable fields can store virtual objects without
forcing them."* **#55 is a prerequisite, not an afterthought.**

**(c) A raw `*mut CelRef` slot array is unrepresentable.** There is no
`raw_load_r` / `raw_store_r` opcode in majit or in rpython; `insns.rs:1095-1102`
registers only `raw_load_i` / `raw_load_f` / `raw_store_i`, and upstream refuses
by name at `jtransform.py:936-937` (`raise Exception("setfield_raw_r not
supported")`). pyre spells the identical structure as a GC array read with
`GETFIELD_GC_R` + `GETARRAYITEM_GC_R` and documents why at
`pyre-object/src/object_array.rs:930-935, :969-987`.

**(d) `Result<CelRef, CelError>` in the traced VM does not lower to an exception
link.** `majit-translate/src/front/result_exc.rs:126` gates the entire
`?`→exception-link lowering on `adt_path_of(err_body, llbc) ==
"pyre_interpreter::error::PyError"`, and the rewrite emits
`CallTarget::method("to_exc_object", Some("PyError"))` (`:541`) /
`("from_exc_object", "PyError")` (`:1372`), with `owner.ends_with(",PyError>")`
at `:1435`. A cel `Result` takes the *other* path:
`codewriter/jtransform.rs:3040-3076` turns each `Ok`/`Err` ctor into `OpKind::New`
+ a `__discriminant` `FieldWrite` — a materialized two-word shell **per VM
operation**, with no known class and no exception link. See §6.3 for the
resolution.

---

## 3. Value representation

```rust
// cel/src/runtime/object.rs
// Port of rclass.py:162-165 OBJECT = GcStruct('object', ('typeptr', CLASSTYPE),
//   hints={'immutable':True,'shouldntbenull':True,'typeptr':True}),
// spelled as pyre-object/src/pyobject.rs:56-69.  ONE word, not pyre's two:
// pyre needs `w_class` for user-defined Python classes; CEL's universe is closed
// except Struct/Opaque, which carry their CEL Type as an ordinary subclass field
// (the `user_overridden_class` shape, baseobjspace.py:40).
#[repr(C)]
pub struct CelObject { pub ob_type: *const CelClass }   // write-once at allocation

#[repr(C)]
pub struct CelClass {
    pub name: &'static str,
    pub kind: CelKind,                 // #[repr(u8)] fieldless enum: coarse family test
    pub gc_type_id: u32,               // == MiniMarkGC::register_type tid == SizeDescr tid
    pub subclassrange_min: i64,
    pub subclassrange_max: i64,        // assigned by freeze_types(), trace.rs:872-883
}
// NO function-pointer slots.  See §2(a).

pub type CelRef = *mut CelObject;
```

Leaves — every one `#[repr(C)]`, **header first**, `align_of <= 8` (const-asserted
by `alloc_with_gc_header`, `majit-gc/src/header.rs:208-213`), **no
`Arc`/`Rc`/`Box`/`Vec`/`String` field**, every managed edge a bare `CelRef` at a
byte offset — the collector traces by offset list, custom hook, or varsize stride
and by nothing else (`majit-gc/src/trace.rs:352, :669-714`):

```rust
#[repr(C)] pub struct W_IntObject       { pub ob_header: CelObject, pub intval: i64 }
#[repr(C)] pub struct W_UIntObject      { pub ob_header: CelObject, pub uintval: u64 }
#[repr(C)] pub struct W_DoubleObject    { pub ob_header: CelObject, pub floatval: f64 }
#[repr(C)] pub struct W_BoolObject      { pub ob_header: CelObject, pub boolval: i64 }
#[repr(C)] pub struct W_NullObject      { pub ob_header: CelObject }
#[repr(C)] pub struct W_BytesObject     { pub ob_header: CelObject, pub length: i64 /* varsize payload */ }
#[repr(C)] pub struct W_StringObject    { pub ob_header: CelObject, pub chars: *mut W_BytesObject, pub length: i64 }
#[repr(C)] pub struct W_ObjArray        { pub ob_header: CelObject, pub length: i64 /* varsize [CelRef] */ }
#[repr(C)] pub struct W_IntColumn       { pub ob_header: CelObject, pub length: i64 /* varsize [i64], NOT ptr-traced */ }
#[repr(C)] pub struct W_TimestampObject { pub ob_header: CelObject, pub nanos: i64, pub off_s: i64 }
#[repr(C)] pub struct W_DurationObject  { pub ob_header: CelObject, pub nanos: i64 }
#[repr(C)] pub struct W_OptionalObject  { pub ob_header: CelObject, pub w_value: CelRef }  // null == none
#[repr(C)] pub struct W_OpaqueObject    { pub ob_header: CelObject, pub w_type: CelRef, pub host_index: i64 }
#[repr(C)] pub struct W_StructObject    { pub ob_header: CelObject, pub w_type: CelRef, pub fields: *mut W_ObjArray }
// The referent of every `w_type` field above, and the result of `type()`.
// `cls` is the CLASS of the type this value denotes -- `type(1)`'s cls is
// `&CEL_INT_CLASS` -- while its own `ob_header.ob_type` is `&CEL_TYPE_CLASS`,
// which is what makes `type(type(1)) == type(string)` true (§6.2).
#[repr(C)] pub struct W_TypeObject      { pub ob_header: CelObject, pub cls: *const CelClass }

// W_ListObject's shape (listobject.py:294-296, :426-433) — one strategy tag +
// one erased storage.  NOT Vec<Box<dyn Val>> (cel list.rs:11).
#[repr(C)] pub struct W_ListObject {
    pub ob_header: CelObject,
    pub strategy:  ListStrategy,       // #[repr(u8)] fieldless enum, not a pointer
    pub storage:   CelRef,             // W_ObjArray | W_IntColumn | W_RecordRows
    pub start: i64, pub length: i64,
}
#[repr(C)] pub struct W_MapObject { pub ob_header: CelObject, pub strategy: MapStrategy, pub storage: CelRef }
```

**`const { assert!(offset_of!(T, ob_header) == 0) }` on every leaf**, emitted by
the same macro that registers the class. `set_vtable_offset(Some(0))` (§7) and
`fuse_boxing_alloc`'s `agg.ob_header` match both depend on it, and neither fails
loudly.

⚠ **`W_TypeObject` was added 2026-08-07 and the omission was internal, not
merely a missing feature.** This enumeration already spent a type value twice —
`W_OpaqueObject.w_type` and `W_StructObject.w_type` are both `CelRef`, and §7
lists both as write-barrier sites — while defining no leaf a `w_type` could point
at. §6.2 independently requires `type` as a builtin, and the spec requires its
result to be a value whose own type is `type`. So the family was load-bearing in
three places and enumerated in none. Found by p99 while implementing `type()`
(task #94).

⚠ **Two different things are spelled `w_type` in this document.** The *field*
above is a `CelRef` — a type **value**, a `W_TypeObject`. The *function* in §4 is
`fn w_type(w: CelRef) -> *const CelClass` — a type **class**, the header word.
They are one indirection apart: `(*w_type_value).cls` is a `*const CelClass`,
`(*any_value).ob_type` is also a `*const CelClass`, and only the former is a
managed object. Do not let the shared spelling collapse them when §3 is
implemented.

**Today `type()` rides `Value::Opaque(Arc<TypeValue>)`**, landed in `90de90b` on
cel-jit — the P2-era encoding, the same one optionals use, and correct for the
tree the plan is actually standing in. `W_TypeObject` replaces it **at P5** with
the rest of the class re-lay; nothing is blocked before then. Recording it here
rather than leaving it to be re-derived: an enumeration that is silently short one
family reads as an oversight later, and this one is referenced by two other leaves
that would have had nowhere to point.

Payload fields are **write-once** and declared immutable so their reads fold to
`getfield_gc_*_pure` (`W_IntObject._immutable_fields_ = ['intval']`,
`intobject.py:543` → `jtransform.py:864-877`). The mechanism already exists:
`#[majit_macros::jit_immutable_fields(intval)]` leaves `_immutable_fields_<Struct>`
marker consts, harvested by `harvest_immutable_fields_from_llbcs`
(`front/llbc_hints.rs:145-165`) into `CallControl::immutable_fields_by_struct`
(`codewriter/call.rs:940-946`), consumed at `codewriter/jtransform.rs:2571-2608`.

Public handle:

```rust
#[repr(transparent)] #[derive(Copy, Clone)]
pub struct Value(pub(crate) CelRef);
// Value::int(i64) / ::bool(b) -> prebuilt TRUE/FALSE / ::null() -> prebuilt NULL
// Value::kind() -> CelKind ; Value::as_int() -> Option<i64> ; …
// NO `unsafe impl Send`.  NO `unsafe impl Sync`.  See §7.
```

---

## 4. Dispatch shape — the narrowing chain

Dispatch is a chain of **pointer-identity tests on the header word**, each arm
calling a concrete monomorphic function. This is `descroperation.py:706-712`
transliterated (`type(w_obj1) is type(w_obj2) and not w_obj1.user_overridden_class`,
then the shortcut) and it is the only shape front-end B lowers to a *direct* call:

```rust
#[inline(always)]
fn w_type(w: CelRef) -> *const CelClass { unsafe { (*w).ob_type } }

fn cel_add(a: CelRef, b: CelRef) -> CelRef {          // error out-of-band, §6.3
    let ta = w_type(a);
    if ta == w_type(b) {
        if ta == &CEL_INT_CLASS    { return w_int_add(a, b);    }   // CallTarget::FunctionPath
        if ta == &CEL_DOUBLE_CLASS { return w_double_add(a, b); }   // -> inline_call
        if ta == &CEL_STRING_CLASS { return w_string_add(a, b); }
        …
    }
    cel_add_slow(a, b)      // deliberately residual; classified per §8/P8
}
```

Three things this buys and the fn-ptr table does not:

1. `(*w).ob_type` becomes a **typed** `FieldRead`: `front/mir.rs:5162-5185`
   [verified] recognises `Projection(Deref)` under a `Field`, resolves the pointee
   class root via `raw_ptr_pointee_class_root`, and inserts a
   `__pyre_cast_instance/<Root>` narrow before emitting the field read.
   ⚠ The design workflow framed this as "a `FieldRead`, not a `RawLoad`". That
   framing is wrong: **majit-translate's `OpKind` has no `RawLoad` variant at
   all** [verified]. The real failure mode is that the narrow does not fire and
   the read is left *classdef-less*, which blocks the annotator downstream.
   **MEASURED — holds** (P0.d assertion 1, `cel_header_read_narrows_to_a_typed_field_read`):
   one narrow, `FieldRead{ name: "ob_type", owner_root: Some("CelObject"),
   owner_id: Some(StructId(..)) }`, zero classdef-less reads.
2. `ta == &CEL_INT_CLASS` is the `rtype_is_` pointer-identity chain pyre already
   depends on (`front/mir.rs:5437-5445`), giving the annotator a knowntypedata
   narrowing.
3. The arm body is a plain `FunctionPath` call, so it inlines when the callee
   graph is in the closure (`pyjitpl.py:2174-2186` /
   `majit-metainterp/src/pyjitpl.rs:14797-14812`) — no `IndirectCallTargets`, no
   `__pyre_wrap_` family, no `bytecode_for_address` miss.

At trace time exactly one arm is taken and the rest become guards — the same
collapse PyPy's `@jit.unroll_safe` if/elif opcode ladder gets
(`pyopcode.py:196-229`).

**Where class facts come from** — in the order they land, *not* guard_class-first:

| source | mechanism | status |
|---|---|---|
| trace-local allocation | `NewWithVtable` → `optimize_new_with_vtable` → `InstancePtrInfo{_known_class, _is_virtual}` | complete: `virtualize.py:207-209`, `optimizeopt/virtualize.rs:543-568, :556` |
| interpreter-proved narrowing | `record_exact_class` → `make_constant_class(update_last_guard=false)` | consumer complete (`optimizeopt/rewrite.rs:2200-2233` → `optimizer.rs:2145-2188`); **front-end-B marker recognizer missing (M3)** |
| non-virtual receiver read | `handle_getfield_typeptr` → `guard_class` | **absent end to end (M4)**; see §8 |

The design does **not** depend on `guard_class`. It depends on `NewWithVtable` +
`record_exact_class`. `guard_class` is a later refinement that shortens
non-virtual receiver chains. **Deviation D11.**

---

## 5. Frame, value stack, and #55

```rust
// cel/src/vm/frame.rs — the pyframe.py:105-112 shape: ONE flat, never-resized
// array laid out | slots | stack |, sized from the code object at compile time.
#[repr(C)]
pub struct CelFrame {
    pub ob_header: CelObject,
    pub last_instr: i64,
    pub valuestackdepth: i64,
    pub locals_stack_w: *mut W_ObjArray,   // a real GC array object.  NOT *mut CelRef.
}
```

`locals_stack_w` is a `W_ObjArray` so its accesses are `GETFIELD_GC_R` +
`GETARRAYITEM_GC_R` (pyre's `object_array.rs:930-935, :969-987`), never
`raw_load_r`, which does not exist (§2c).

`CelFrame` is declared **virtualizable** with `locals_stack_w[*]` in the array
list. Under a virtualizable, slot reads and writes never become trace operations
at all — `opimpl_getfield_vable_r` / `_opimpl_setfield_vable` and
`getarrayitem_vable` / `setarrayitem_vable` read and write
`metainterp.virtualizable_boxes[index]` in the tracer (`pyjitpl.py:1166-1199`,
`:1218-1251`). Nothing is emitted, so nothing forces the value parked there. That
is the entire mechanism by which a freshly built `W_IntObject` survives a push as
a virtual.

Contract obligations (`virtualizable.rst:44-81`): fixed size, never resized,
never reassigned, never passed around, only `frame.locals_stack_w[i]` direct
access, `i` provably non-negative. Promote the depth right after the merge point,
as PyPy does: `self.valuestackdepth = hint(self.valuestackdepth, promote=True)`
(`interp_jit.py:82-91`).

Price of the escape, to be budgeted not discovered: `gen_store_back_in_vable`
writes back every static slot **and** every array element at once
(`pyjitpl.py:3489-3521`).

Known debt: majit's `PtrInfo::Virtualizable` has no upstream counterpart in
`optimizeopt/info.py` and its own comment marks it pyre-specific with a
convergence TODO (`optimizeopt/virtualize.rs:397-400`). Upstream's mechanism
lives in `rpython/jit/metainterp/virtualizable.py:10 VirtualizableInfo` — the
tracer layer, not the optimizer. Any cel use inherits that debt; record it, do
not extend it.

---

## 6. Portal, code object, and the two things §2 rules out of it

### 6.1 Portal and JitDriverSpec

```rust
// cel/src/vm/portal.rs  — a FREE function.
pub fn cel_eval_loop(next_instr: i64, code: *const CelCode, frame: *mut CelFrame) -> CelRef;
```

`Program::execute(&self, ctx: &Context)` stays a thin `&self` wrapper that builds
the frame and calls it (`cel/src/lib.rs:180-194` today).

The portal must be free **not** for the `--start-from` reason in the task brief —
extraction is whole-crate with `--include` and `scripts/llbc_extract.py:33-34`
documents no `--start-from`. The real reason: warmspot derives
`_PORTAL_FUNCTYPE = FuncType(greens ++ reds, RESTYPE)` from the jit_merge_point
argument list and splits the graph at the marker (`warmspot.py:419-431, :658-671`);
an `&self` receiver is neither a green nor a red.

```rust
JitDriverSpec {
    portal:         CallPath::from_segments(["vm", "portal", "cel_eval_loop"]),
    greens:         vec!["next_instr".into(), "code".into()],   // INT then REF, jit.py:665-681
    reds:           vec!["frame".into()],
    autoreds:       false,
    virtualizables: vec!["frame".into()],                        // §5 — mandatory
    red_types:      vec!["CelFrame".into()],
}
```

`jit_merge_point!()` is the **first statement** of `loop {}`, before the opcode
decode and before any code-object field is read. `can_enter_jit` goes **only** on
the backward branch of the comprehension jump, in the jump handler, with nothing
between it and the following merge point (`interp_jit.py:101-120`,
`jit.py:746-754`).

⚠ **Corrected 2026-08-07.** An earlier draft said "do not take the rsre
no-`can_enter_jit` shape" because "nothing sets `no_loop_header` true". That
reached the right conclusion for the wrong reason, twice over. (a) rsre's
single-activation driver `jitdriver_Match` (`rsre_core.py:1382-1390`, greens
`['pattern']`, merge point at `:1389` with no surrounding loop and no
`can_enter_jit`) does **not** work through `no_loop_header` — it works through
the FINISH-from-`ResumeFromInterpDescr` path in P0.f. `no_loop_header`
(`warmspot.py:766/775/790`) only relaxes *when* `opimpl_jit_merge_point` may
auto-stamp a loop header (`pyjitpl.py:1546-1556`). (b) majit's `no_loop_header`
is a `pub` field with a ported consumer (`pyjitpl/dispatch.rs:5036-5106`) and an
in-tree test that sets it true (`dispatch.rs:10447`) — "nothing sets it true" is
true only of production. The conclusion stands: CEL's comprehension back edge is
a real loop and gets a real `can_enter_jit`. The *other* door — a compiled
procedure for a comprehension-free expression — is P0.f's subject and needs no
`can_enter_jit` at all.

The marker receiver type name must be added to `RECOGNIZED_JITDRIVER_RECEIVER_ROOTS`
(`codewriter/jtransform.rs:338-339`, currently exactly
`["PyPyJitDriver","UnpackIterableJitDriver"]`) — that is **M2**.

**Warm-entry wiring is unassigned in the current front-end-A shape and must be
named.** cel's mainloop takes the driver as an argument
(`cel/src/majit/bytecode.rs:893 driver: &mut majit_metainterp::JitDriver<VmStateF>`)
and stores drivers in a `HashMap<(usize, usize, u32), JitDriver<VmStateF>>`
(`:1939`) — those are #53's caches. The `#[jit_interp]` macro generates the
JitCell/threshold/`ContinueRunningNormally` wiring today; a front-end-B cel has to
write it. Good news: `warmstate.rs` and `warmspot.rs` live in **majit-metainterp**,
not pyre, so the warm-entry layer is already interpreter-neutral and #84 is
narrower than "the runtime". P8 deliverable, scoped by P0.c.

### 6.2 The code object — and host functions, which are §2(a) again

```rust
pub struct CelCode {
    pub co_code:     Box<[i32]>,          // raw/immortal, NOT traced (see below)
    pub co_consts:   *mut W_ObjArray,     // a GC object, registered as a prebuilt root
    pub co_names:    Box<[Box<str>]>,     // resolved to slot INDEX at compile time
    pub co_builtins: Box<[u16]>,          // builtin OPCODE ids — not fn pointers
    pub co_nslots:   u32,
    pub co_stacksize: u32,
}
```

`co_consts` is a `W_ObjArray`, not `Box<[CelRef]>`: a Rust container holding
managed pointers is memory the collector does not trace. pyre sidesteps the same
problem by holding the code body behind an opaque `code_ptr: *const ()`
(`pyre-interpreter/src/pycode.rs:150-153`). The other three arrays are read-only
and never hold a `CelRef`, so they stay plain Rust and are read outside or
promoted.

**Host functions do NOT go in a table.** `cel/src/magic.rs:301` is
`pub type Function = Box<dyn Fn(&mut FunctionContext) -> ResolveResult + Send + Sync>`
in a `FunctionRegistry { functions: BTreeMap<String, Function> }`
(`magic.rs:281-283`). Calling one from inside the traced loop is exactly the
defect §2(a) rules out: `dyn Fn` → `CallClass::Dynamic` → the `__dyn_call`
placeholder that stops the graph; a bare fn pointer → `CallKind::Ptr` →
`LowerError::Unsupported`. So the story splits:

* **CEL builtins** (`size`, `startsWith`, `contains`, `matches`, `type`, `has`,
  the macros) become **integer opcodes** with a compile-time `match` ladder of
  `FunctionPath` calls. No table, no indirection, inlinable.
* **User-registered functions** get a *deliberate* residual path that actually
  lowers — which today **does not exist**. That is **M8**, and it is required
  before P8, not inside it.

### 6.3 Errors are out-of-band; the traced VM never uses `?`

Per §2(d), `Result<CelRef, CelError>` materializes a two-word shell per VM
operation. Two ways out; **take the second**:

* M7: parameterize `tyref_is_result_of_pyerror` / `owner_is_result_of_pyerror` /
  the `to_exc_object`/`from_exc_object` method names off driver config. Larger
  than M4, and it buys cel a machine it does not need.
* **Chosen:** carry CEL errors out-of-band — a per-heap `last_error: CelRef` slot
  plus a sentinel `CelRef` return — which is what PyPy actually does
  (`OperationError` on the ExecutionContext, not a return-value union). **Rule:
  no `?` and no `Result` anywhere reachable from the portal.** The `Result` API
  is reconstructed at the `Program::execute` boundary, outside the traced graph.

`ExecutionError` (`cel/src/lib.rs:68-137`, eleven `Value`-carrying variants,
`Clone + PartialEq`) stays exactly as it is at the public boundary.

### 6.4 The bind step — `Program` alone cannot intern a code object

`Program::compile(source)` takes no context, while
`Context::Root { functions, variables, resolver, env }` (`cel/src/context.rs:36-47`)
is per-execution and has a **dynamic** `VariableResolver` escape hatch. §6.1
requires `co_names` resolved to slot indices at compile time *and* the code object
never rebuilt per evaluation (REF greens are keyed by pointer identity,
`warmstate.py:108-128`). Those cannot both hold when the name set is known only at
bind time.

So introduce the link step explicitly:

```rust
pub struct BoundProgram { code: Arc<CelCode>, /* interned, one per (Program, Context shape) */ }
impl Program { pub fn bind(&self, ctx: &Context) -> BoundProgram; }
```

Cache key = (`Program` identity, the sorted name set + arity signature of the
context). A `VariableResolver`-supplied name that is absent at bind time compiles
to a **late-lookup opcode** that is residual, not to a slot. State this in the API
docs: the same `Program` meeting a second `Context` shape mints a second
`BoundProgram`, and the green identity is the code object, never the `Program`.

The current per-evaluation string chain must not survive:
`ctx.env().find_overload(&call.func_name, &args)`, `ctx.get_function(name)`
walking the `Context::Child` parent chain, `format!("{prefix}.{}")`, plus a
`Vec<Cow<dyn Val>>` per call (`objects.rs:1896-1930`).

---

## 7. Allocation, heap, write barriers, roots, threading

**Allocation entry point — by value, one argument.** `fuse_boxing_alloc` gates on
`!is_malloc_typed(target) || args.len() != 1` (`model.rs:2934`), requires that one
argument to be the result of a `SyntheticTransparentCtor` for a struct in
`struct_field_attrs` (`:2942-2957`), requires a `FieldWrite{base:%agg,…}` for
every non-`ob_header` field *before* the call (`:2967-2991`), and resolves the
vtable through `agg.ob_header` → `header.ob_type` →
`__pyre_cast_instance(ConstRefAddr(t))` (`:2906-2911`), declining with a bare
`continue` if that address is zero (`:3009-3011`). pyre's real signature is
`pub fn malloc_typed<T: GcType>(value: T) -> *mut T` (`pyre-object/src/lltype.rs:256`,
called by value at `intobject.rs:106-122`).

```rust
// cel/src/runtime/lltype.rs — path segments must end `lltype::malloc_typed`
// or `lltype::malloc_typed_managed` (model.rs:2851-2859 is_malloc_typed).
pub fn malloc_typed<T: CelGcType>(value: T) -> *mut T;
pub fn malloc_typed_managed<T: CelGcType>(value: T) -> *mut T;

let w = lltype::malloc_typed(W_IntObject {
    ob_header: CelObject { ob_type: &CEL_INT_CLASS },
    intval: x,
});
```

A nullary `malloc_typed::<T>()` with alloc-then-init stores through the returned
pointer matches **nothing** and produces zero `NewWithVtable` with no error.
`model.rs:2784-2788` names alloc-then-init as the form pyre deliberately does not
use.

**Class-static addresses — M1.** `resolve_vtable_addr` reaches the type pointer
through the driver-supplied `HostStaticAddrs.pytypes` bucket (`lib.rs:463-471`),
but the narrow it emits hard-codes the root name:
`CallTarget::FunctionPath{segments:["__pyre_cast_instance","PyType"]}` with
`result_ty: ValueType::Ref(Some("PyType"))` (`front/mir.rs:5445-5471`), and
`pytype_static_addr`'s own doc says "the root is always `PyType`"
(`mir.rs:6168-6187`). A `CelClass` static cannot union with cel's `ob_type` field
cell through that bucket.

**Prebuilt singletons must carry a GC header.** Do **not** use plain Rust
`static`s for TRUE/FALSE/NULL/small-ints. pyre holds its singletons in a
`OnceLock<usize>` filled by `malloc_typed_immortal`
(`pyre-object/src/lltype.rs:282-292` → `alloc_with_gc_header_immortal`), precisely
so they carry a header — `boolobject.rs:59-65`, `noneobject.rs:26`.
`majit-gc/src/header.rs:188-196` narrows the immortal path to *pointer-free leaf*
payloads, so any prebuilt holding a `CelRef` (including `co_consts`) needs
prebuilt-root registration instead (`prebuilt_root_objects`,
`majit-gc/src/collector.rs:494-497`, porting `incminimark.py:355`).

**Write barriers — one per mutable ref field, no exceptions.**
`majit-gc/src/header.rs:188-196` states the failure mode directly: an ordinary
boxed object does not qualify for the immortal path because *"its reference fields
are written at construction with no write barrier, so once the first barrier on
any other field opens the object, a major would follow those never-updated slots
into freed memory."* `majit-gc` exposes `write_barrier` / `write_barrier_managed`
on the collector trait (`majit-gc/src/lib.rs:550, :557`); 19 files under
`pyre/pyre-object/src/` call it. Barrier sites in this design:
`W_OptionalObject.w_value`, `W_ListObject.storage`, `W_MapObject.storage`,
`W_StructObject.fields`, `W_StructObject.w_type`, `W_OpaqueObject.w_type`,
`CelFrame.locals_stack_w`, and every `setarrayitem` into a **non-virtual**
`W_ObjArray`. Enforce by construction: no `CelRef` field is ever written except
through a `set_ref` helper, checked by an audit test (P6 tripwire).

**The root set has an owner and a phase.** Non-moving old-gen removes the *move*
hazard, not the *free* hazard. pyre's rationale
(`pyre-object/src/gc_interp.rs:1-35`) puts the safepoint at loop top *"where the
only live refs are in the frame and reachable through the registered `pyframe`
root walker."* cel has no such walker. The root set is:

1. the live `CelFrame` and its `W_ObjArray`,
2. `Context` variables,
3. `CelCode.co_consts` (prebuilt-root registered),
4. the in-flight result and the eleven `ExecutionError` `Value` payloads,
5. **an embedder handle registry** — `Program::execute` returns an unlifetimed
   `Value` the embedder may hold across the next `execute` on the same heap, and
   with `Arc` gone nothing keeps it alive.

Item 5 means the public `Value` becomes a **rooted handle**. That is a larger API
break than `!Send` and it is owned by **P5**, not discovered at P6.

**Backend class-identity mode: `set_vtable_offset(Some(0))`.** With `Some(ofs)`
the assembler emits `cmp QWORD [Rq(obj) + ofs], classptr` — a plain in-object read
(`majit-backend-dynasm/src/x86/assembler.rs:4753-4788`). With `None` it falls to
`_cmp_guard_gc_type`, whose `emit_load_gc_typeid_into_reg` is
`mov Rd(dst), [Rq(obj) - GcHeader::SIZE]` (`:4805-4808`). Since `fuse_boxing_alloc`
requires the in-object `ob_type` word anyway, `Some(0)` is the coherent choice.
Do not mix the two models. (`emit_guard_is_object` / `emit_guard_subclass` at
`:4841-4900` index `base_type_info + tid*sizeof_ti` off the header regardless —
another reason every prebuilt needs a real header.)

**Heap.** One `CelHeap` per thread. Until cel's root walker exists, keep values on
the **non-moving** old-gen and drive an old-gen-only major at a VM-dispatch
safepoint — pyre's stepping stone and its stated reason
(`pyre-object/src/gc_interp.rs:11-16, :267-286`). Do **not** ship a
dropped-at-return bump arena: `Program::execute` returns an unlifetimed `Value`
and eleven `ExecutionError` variants carry an owned `Value`, both of which would
dangle. Do **not** allocate from inside the metainterp's tracing path: its
register bank is not a root set and MiniMarkGC's headerless entry panics by
contract (`majit-gc/src/lib.rs:320-346`).

**`W_OpaqueObject` reclamation.** Moving the host object behind `host_index` into
a per-heap side table removes the `Arc<dyn Opaque>` that freed it for nothing.
Either register a finalizer (and pay its cost) or adopt the contract *"opaque host
objects live as long as the heap"* — **decide at P5 and record it as D12.**

**Threading — the honest break.** `Value` is `*mut CelObject` and is **not**
`Send` and **not** `Sync`. Do not write the `unsafe impl`; let the compiler
enforce it. Consequences, all semver-major and all deliberate:

- `#[test] fn test_context_is_send()` (`cel/src/context.rs:301-310`) is deleted;
  `Context` becomes `!Send`.
- `Val: Any + Debug + Send + Sync` (`common/value.rs:9`) is gone with `dyn Val`.
- `Opaque: Any + OpaqueEq + AsDebug + Send + Sync` (`objects.rs:652`) becomes
  `!Send`.
- `Function = Box<dyn Fn(&mut FunctionContext) -> ResolveResult + Send + Sync>`
  (`magic.rs:301`) and `VariableResolver: Send + Sync` (`context.rs:280`) lose the
  bounds.
- `example/src/threads.rs:5-22` (one `Program` shared across `std::thread::scope`)
  becomes one `Context` and one heap per thread. `Program`/`CelCode` stay `Sync`
  because `co_consts` holds only immortal prebuilts and prebuilt roots.
- Escape hatch: `Value::detach() -> OwnedValue` (deep copy to a plain owned tree)
  and `OwnedValue::attach(&CelHeap) -> Value`.

The alternative — `unsafe impl Send + Sync` — is what pyre does, justified by the
GIL (`majit-gc/src/lib.rs:1002-1009`: *"Exclusion comes from the GIL (`rgil`),
which the caller holds"*). **cel has no GIL. Do not inherit the impl without the
GIL.**

---

## 8. The majit change budget

None of these is config. All are PRs against the pyre repo, each followed by a cel
`Cargo.toml` rev bump — cel pins majit by git rev with an explicit in-file warning
that a stale pin is **silent** (`cel/Cargo.toml:33-43` [verified]). **Every M row
carries a rev-bump + re-census step**, and M1–M8 land on one named pyre branch
agreed with the user before P5.

| # | change | size | blocks |
|---|---|---|---|
| M1 | Parameterize the `__pyre_cast_instance` class-root name off `"PyType"` (`front/mir.rs:5445-5471`, `:6168-6187`) so `HostStaticAddrs.pytypes` can carry `CelClass` statics. **MEASURED at P0.d: this is the sole gate on the boxing fuse — without it `fuse_boxing_alloc` returns 0 in silence; with the address as a constant the cluster fuses.** | small | P5 |
| M2 | Add the cel driver receiver to `RECOGNIZED_JITDRIVER_RECEIVER_ROOTS` (`codewriter/jtransform.rs:339`) | 1 line | P8 |
| M3 | `record_exact_class` front-end-B marker recognizer beside `jit_promote_marker` (`front/mir.rs:10249-10258`), emitting the existing `BC_RECORD_EXACT_CLASS = 201` (`codewriter/insns.rs:549-556`). Consumer already complete end to end (`jit.rs:801`, `jitcode/assembler.rs:2964`, `blackhole.rs:6645` wired at `:8927`, `trace_ctx.rs:3949`, `optimizeopt/rewrite.rs:2200`) | ~30 lines | P6 |
| M5 | Varsize allocation lowering (`NEW_ARRAY` / `NEW_ARRAY_CLEAR`) for `W_ObjArray` / `W_BytesObject`. `fuse_boxing_alloc` handles a fixed-size cluster only (`model.rs:2845-2911`); the two `OpKind::New` producers are the tuple arm (`jtransform.rs:3011-3023`) and the `Result` arm (`:3038-3076`) | medium | P6 lists/strings |
| **M8** | A user-registered-function residual call path that actually lowers (§6.2). Today `dyn Fn` → `__dyn_call` placeholder, fn ptr → `LowerError::Unsupported` | **medium–large** | **P8** |
| M6 | #84 — whatever P0.c says is actually missing | unknown | P8 |
| M4 | **`guard_class` full vertical slice** (below) | large | **nothing on the critical path — §4** |
| M7 | Parameterize the `PyError` Result-lowering hardcode | large | **not taken** — §6.3 routes errors out-of-band instead |

**M4 in full, so nobody under-sizes it later:** a `BC_GUARD_CLASS` insn number +
argcode, a `jitcode/assembler.rs` emitter, a generic `opimpl_guard_class` in
`pyjitpl/dispatch.rs` returning the class and gated on heapcache
`is_class_known`/`class_now_known` (`majit-trace/src/heapcache.rs:1010-1018`,
already exists), a `bhimpl_guard_class` over an interpreter-neutral `cls_of_box`,
plus resume/liveness — then the front-end port of `is_typeptr_getset`
(`jtransform.py:952-954`), `handle_getfield_typeptr` (`jtransform.py:1004-1010`)
and the typeptr-setfield drop (`jtransform.py:906-909`). Today the only GuardClass
emissions in-tree are pyre's hand-rolled exception path
(`pyjitpl/dispatch.rs:7720-7770`) and `box_trace.rs:113, :328`.

**#84 is very likely mis-sized and must be re-scoped before anything
irreversible.** `majit-metainterp/src/jitcode/mod.rs:15` is literally
`pub use majit_translate::insns;`, and `majit-metainterp/src/pyjitpl/dispatch.rs`
dispatches 188 distinct `BC_*` constants off that table — including the whole
`BC_INLINE_CALL` / `BC_INLINE_CALL_{R,IR,IRF}_*` family (`:5732`, `:5822-5831`),
which is the front-end-B graph-closure call shape — with a real
`run_to_end(&mut self, ctx: &mut TraceCtx, …)` calling `ctx.record_op`.
`pyre-jit-trace/src/jitcode_dispatch` contains **zero** `BC_` constants; it is a
second, name-string-keyed walker (`"record_exact_class/ri" => …`, `mod.rs:10053`)
that calls itself "the sole production tracer" (`mod.rs:8`). Also in cel's favour:
`warmstate.rs` / `warmspot.rs` are already in majit-metainterp. **P0.c decides
whether #84 is a 59k-line lift or a gap list.**

---

## 9. Phases

### P0 — instruments and six probes. Nothing irreversible. Go/no-go for the plan.

**P0.a — the runtime allocation counter** (primary gate for every data-model
phase). cel's existing harness is **not** what it needs to be:
`cel/examples/allocs.rs` is a `#[global_allocator]` *example binary* wired to the
columnar batch API (`use cel::majit::batch::{Batch, BatchProgram, ColumnRef, Tier}`,
`ROWS = 50_000`) — the tier P9 retires. A global allocator promoted into
`cargo test` counts the harness's own allocations and races across the default
multi-threaded runner. So build: one integration test with `harness = false` (or
`--test-threads=1`), a thread-local counter, **allocations per evaluation** over a
fixed corpus (cometkim per-call set + a list-binding case + a comprehension case +
the register machine's own cases), and a **per-host** blessed baseline file. From
P5 on, add a **second** counter inside `CelHeap::malloc_typed`: a bump/nursery
allocation is invisible to `GlobalAlloc` by construction.

> **Drop terminators are a SECONDARY counter, never the gate.** `*mut CelObject`
> has no drop glue, so the walker's 1248 collapses toward 0 the instant
> `Box<dyn Val>` becomes `CelRef` regardless of what the heap does. Front-end B
> lowers `TermKind::Drop` to a bare goto anyway (`front/mir.rs:6465-6470`), so the
> count is a proxy for owning places, not for emitted forcing ops.

**P0.b — reproduce the static census. Its first commit is the cel extraction
path, because none exists** [verified]: `pyre/scripts/extract-llbc.py:26 SPECS`
holds only `corpus`, `pyre-object`, `pyre-module`, `pyre-interpreter`, `pyre-jit`;
`build/llbc/` has no cel artefact; cel has no `build.rs`; and cel does **not**
depend on `majit-translate` at all.

⚠ **The run below was real and its numbers are measured — but it was ad-hoc and
uncommitted, so nobody reading this document could reproduce it (#107).** That
is *unreproducible*, not *unsourced*; do not read the paragraph as unsupported.
A committed producer now exists —
`majit/majit-translate/tests/test_cel_census.rs` in the pyre-wasmi tree, run as

```sh
CEL_CENSUS_LLBC=cel-jit/build/llbc/cel-portals.ullbc \
  cargo test --release -p majit-translate --test test_cel_census -- --nocapture --test-threads=1
```

⛔ No `--ignored`: the `ignore` is `cfg_attr(debug_assertions, …)`, so a release
run with `--ignored` selects **nothing** and prints `running 0 tests` — which
reads exactly like a pass.

**RUN 2026-08-07 at "skills: correct the cel design's back-edge premise, which
is refuted" (`492ac152a8d` as of 2026-08-11 — branch-local; the sha rots at each
rebase, so re-derive it from that subject) — GATE PASS.** All ten §1 cells reproduced
exactly, on four independent artifacts, both opt levels. §1's blank *clean*
cells, now filled: 2031 ops, 239 : 12, `guard_class` 0, `vtablemethodptr` 0. The
36 %-real-computation row reproduces for mainloop: `binop` 328 + `arrayread` 285
+ `arraywrite` 69 = 682 / 1873. Five corrections to this paragraph, all measured:

1. ⛔ **The extraction path does NOT go in the pyre driver.** `scripts/extract-llbc.py`
   forwards to `pyre/scripts/extract-llbc.py`, whose engine fingerprints sources
   via `git ls-files` against its own root — and `git ls-files -- cel-jit`
   returns **0 files**, because cel-jit is a separate repo, not a submodule.
   Empty set ⇒ constant stamp ⇒ the skip logic never re-extracts ⇒ **silently
   stale LLBC.** Built instead: `cel-jit/scripts/extract-llbc.py`, a per-repo
   driver over the *neutral* engine, importing `CrateSpec`/`run_cli` from
   `pyre/scripts/llbc_extract.py` via `PYRE_ROOT`. **Zero engine changes.**
2. ⭐ **`--opaque cel::parser` is what defeats the charon SIGABRT.** Whole-crate
   charon aborts with `could not find region '1_0` → rustc stack overflow →
   signal 6, exit 101; `RUST_MIN_STACK` does not help. The sole cause is the
   ANTLR-generated `cel/src/parser/parser.rs`. Marking it opaque is free — no
   portal closure calls the parser. The needed shape is **whole-crate with
   exactly one opaque module**, not `--include` and not `--start-from`.
3. ⛔ **The `--include` vs `--start-from` tension (line 347 below) is a false
   dilemma.** `llbc_extract.py:33-34` merely *exemplifies* `--include`;
   `charon_args` is a free-form flag list and has always accepted `--start-from`.
4. ⛔ **"Check in `cel.ullbc` + fingerprint" contradicts convention** — the
   superproject gitignores `/build/`, pyre tracks 0 files under `build/llbc`,
   and the artifact is 100 MB.
5. ⛔ `resolve_val` resolves **crate-stripped** (`objects::Value::resolve_val`);
   `cel::objects::Value::resolve_val` names no graph. The two free functions
   resolve crate-included. Cause (inferred): `free_function_alias_paths`
   (`majit-translate/src/lib.rs:583-637`) widens *free-function* spellings with
   `local_crate_roots()`; inherent-impl methods get no such widening.

⛔⛔ **PIN THE ARTIFACT SHAPE — scope changes the census.** `run_mainloop_f` on
whole-crate `cel.ullbc` gives 16 jitcodes / 2048 ops / 263 : 20 / `new_ops` 4;
on the `--start-from`-scoped `cel-portals.ullbc`, 13 / 1873 / 239 : 20 / 0. A
re-extracted whole-crate control under the worktree majit gave **identical
divergent numbers**, so the cause is scope, not majit provenance. **§1 is
reproducible only against the scoped artifact.** Whole-crate consumption is also
impractical: `clean_interp_seeded_f` on `cel.ullbc` passed **8.4 GB RSS** and
began swapping, mainloop peaks ~3 GB / ~2 min, versus **4.4–8.5 s** for the same
portals on the scoped artifact. Everything iterative in P1+ uses the scoped one.

**The three derived counters** — `dangling_vtablemethodptr` and `new_ops` are
measured: `new_ops` walker **65** / clean 1 / mainloop 0, with
`new_with_vtable` **0 everywhere** (all 65 are bare `new`);
`dangling_vtablemethodptr` walker **49** / 0 / 0. ⛔ The op prints as
`vtablemethodptr` in `JitCode::dump()` while the *insns-table* key is
`vtable_method_ptr/rd>i` — matching only the latter silently reads 0.

⛔ **`indirect_targets_attached` is NOT reachable, and that is a finding.**
`IndirectCallTargets` is produced at one site (`codewriter/jtransform.rs:5199`)
and **merged into a single global identity set** at assembly
(`codewriter/assembler.rs:1332-1341`); the pipeline surfaces only
`indirectcalltarget_indices` (`lib.rs:2063-2070`). The merge destroys the per-op
attachment — you can see which functions are indirect-call targets, never which
`residual_call` ops carry a sidecar. Getting the real counter needs a
majit-translate change. Two **named proxies**, not substitutes:
`indirectcalltarget_indices.len()` = 11 / 0 / 0, and `guardvalue`-then-
`residual_call*` = 19 / 0 / 0.

⛔ `functions` is empty on every run (`lib.rs:1868` hardcodes `Vec::new()`), so
"0 transform notes" is never evidence of clean lowering.

**P4 is therefore STARTED at P0, not run in parallel from it.**
*Gate:* the §1 table reproduces — **PASSED**.

**P0.c — the tracer probe (#84 re-scope).** Feed a front-end-B-produced cel
jitcode set to the generic `JitCodeMachine` (`pyjitpl/dispatch.rs:1166-1213`).
Record exactly which opcodes it cannot handle, and what warm-entry wiring is
missing (§6.1). **Capture the run as a golden file — it is P7's fixture set.**
Output: a gap list, or a confirmation that #84 is a real 59k-line lift.

**P0.d — the lowering fixture. THIS IS THE EXECUTABLE FIRST STEP, and its home
already exists** [verified]: `majit/charon-corpus/` — a workspace-isolated
micro-crate (`[workspace]` in its `Cargo.toml` keeps it out of the parent build),
a checked-in `corpus.ullbc`, `inspect_llbc.py`, and a README pinning Charon
`nightly-2026.05.29` via `scripts/install-charon.py`. It is already `SPECS["corpus"]`.
So P0.d is **"add four functions to `charon-corpus/src/lib.rs` and assert with
`inspect_llbc.py`"**, not "build a harness":

1. `(*w).ob_type` where `w: *mut CelObject` lowers to a **typed `FieldRead`** with
   a `__pyre_cast_instance` narrow (`front/mir.rs:5162-5185`).
2. `lltype::malloc_typed(W_IntObject{ ob_header: CelObject{ ob_type: &CLS }, intval: 1 })`
   fuses to **`NewWithVtable` with a nonzero vtable** plus payload `FieldWrite`s
   (`model.rs:2934-3045`).
3. `if ta == &CLS { concrete_fn(a,b) }` lowers the arm to a **`FunctionPath`**
   call, and the callee appears in the graph closure.
4. the `_immutable_fields_<Struct>` marker makes the payload read
   `getfield_gc_i_pure`.

**RUN 2026-08-07. Fixture: `majit/charon-corpus/src/lib.rs` §8 (`cel_w_type`,
`cel_new_int`, `cel_add`, `w_int_add`, `lltype::malloc_typed`, the two class
statics, `_immutable_fields_W_IntObject`). Assertions:
`majit-translate/tests/test_mir_frontend.rs`. Corpus re-extracted,
`has_errors: false`; `majit-charon-reader` + `majit-translate` suites fully
green (the reader's local-fn count moved 12 → 20 — `static`/`const` initializer
bodies count as functions).**

| assertion | result |
|---|---|
| 1 header read | ✅ **holds** — one narrow, `owner_root: Some("CelObject")`, `owner_id: Some(..)`, zero classdef-less reads |
| 2 boxing cluster | ⚠ **conditional** — see below |
| 3 narrowing arm | ✅ **holds** — `Call FunctionPath ["charon_corpus","w_int_add"]`, zero `__dyn_call`, 2 header reads + 2 `eq` (the `type(a) is type(b)` test then the per-class shortcut) |
| 4 pure payload read | ⛔ **not checkable at this layer** — `harvest_immutable_fields_from_llbcs` feeds `CallControl`, consumed by the codewriter (`codewriter/jtransform.rs:8458`) and `layout.rs`, both downstream of `lower_function`. `FieldRead.pure` is `false` here by construction. Needs a codewriter-level test; the marker is in the corpus ready for it. |

**Assertion 2 in full, because it is the one that re-sizes the plan.** Every
structural part of the cluster already matches: one by-value `malloc_typed`
argument, a `SyntheticTransparentCtor` aggregate, the nested
`CelObject`→`W_IntObject` header chain, and the payload `FieldWrite`. The single
missing input is the **class-static's address**: `&CEL_INT_CLASS` lowers to
`Call FunctionPath ["charon_corpus","CEL_INT_CLASS"]`, and `resolve_vtable_addr`
accepts only `ConstRefAddr` or a `__pyre_cast_instance` walk over one
(`model.rs:2884-2911`), so it returns 0 and `fuse_boxing_alloc` declines with a
bare `continue` — **0 fused, no diagnostic**. Substituting a `ConstRefAddr` for
that one read makes the cluster fuse: **1 `NewWithVtable{ owner: "W_IntObject",
vtable: <addr> }`**, payload store re-emitted, `malloc_typed` consumed. The
decline's own comment names this case exactly (`HostStaticAddrs.pytypes` empty).

⇒ **M1 is confirmed as the gate, and confirmed to be the only one.** It is not
optional and it is not deferrable past P5. The test pins both halves so a
regression in either is loud.

**P0.e — re-verify the twelve load-bearing citations** (§13) **by symbol, not by
line**, and record the drift.

**P0.f — RAN 2026-08-07, and it REFUTED its own premise.**

⛔ **"A CEL program without a comprehension has no back edge and can never trace"
is FALSE.** A loop-free trace is compiled upstream as a *procedure attached to
the interpreter entry*, not as a loop:

`MetaInterp.finishframe` → `compile_done_with_this_frame`
(`rpython/jit/metainterp/pyjitpl.py:3222-3244` [verified]) records `FINISH` and
calls `compile_trace(self, self.resumekey, exits)`. For a trace started from the
interpreter the resumekey is `ResumeFromInterpDescr`. In `compile_trace`, on
`info.final()`, `target_token = new_trace.operations[-1].getdescr()` — **the last
op is the `FINISH` just recorded, so its descr *is* `token` and the
`if target_token is not token: giveup()` guard does not fire** [verified,
`compile.py:1080-1083`]. `ResumeFromInterpDescr.compile_and_attach`
(`compile.py:1006-1022` [verified]) then sends the trace to the backend and calls
`warmstate.attach_procedure_to_interp(original_greenkey, jitcell_token)` — its own
comment: *"send the new_loop to warmspot.py, to be called directly the next
time."*

**majit ports every link** [verified]: `attach_procedure_to_interp` at
`majit-metainterp/src/pyjitpl.rs:10271`, `:10299`, `:21110`, `:21383`;
`ResumeFromInterpDescr.compile_and_attach` parity at `:11277`, `:11615`.

For a `(next_instr, code)` driver the entry green key is `(0, code)` — stable per
`BoundProgram`, which §6.4 already requires. **That is exactly the
comprehension-free predicate case.**

⇒ The census does not decide P6–P8. It only bounds the downside *if the back edge
were the only door*, and it is not.

**But #83's counter is not lying.** `finish_and_compile` fires `on_compile_loop`
(`pyjitpl.rs:8900`), and cel's `COMPILES` *is* that hook
(`cel/src/majit/bytecode.rs:1857-1858`), so a FINISH-terminated procedure **would
have been counted**. `COMPILES == 0` on 16 cases therefore means nothing compiled
at all — **a behaviour defect to diagnose, not a structural impossibility.**

### #83 RCA — CLOSED 2026-08-07, by call graph, no instrumentation needed

The three-way triage above resolved to the **first** branch, and the reason is
structural: **the `#[jit_interp]` portal has no portal-entry warm-up door.**

Upstream has **two** doors, and front-end A opens only one:

| door | upstream | `#[jit_interp]` |
|---|---|---|
| portal **entry** | `warmspot.py:941-945` `ll_portal_runner` → `maybe_compile_and_run(state.increment_function_threshold, *args)`, comment `# maybe enter from the function's start.` | ⛔ **absent** |
| **back edge** | `warmspot.py:576-579` `maybe_enter_jit` → `maybe_compile_and_run(state.increment_threshold, *args)` | ✅ `can_enter_jit!` → `can_enter_jit_keyed` (`jitdriver.rs:4099`) → `back_edge_internal` |

The entry-door port is written, complete, and **unused**:
`majit-metainterp/src/warmstate.rs:1407 should_trace_function_entry`, over
`increment_function_threshold` (`:335`, `:453`, `:1210`), with the
`warmstate.py:482-511` decline gates ported. Repo-wide it has exactly **one**
production caller — `pyre/pyre-jit/src/eval.rs:9278`, pyre's hand-written portal.
Every other hit is a unit test or a doc comment.

`jit_merge_point!()` cannot substitute: it expands
(`majit-macros/src/jit_interp/mod.rs:2093-2130`) to a body wrapped in
`if #driver.is_tracing() { … }` — a **no-op when not tracing**, so it can never
*start* tracing. It is a merge point, not a door.

cel calls `can_enter_jit!` in exactly one place,
`cel/src/majit/bytecode.rs:1408-1410`, guarded by `if tgt < pc` — a **backward**
jump. A CEL program with no backward jump therefore ticks no counter at any pc on
any number of calls. Branches (b) trace-aborted and (c) `giveup()` are ruled out
by construction: tracing never begins.

⛔ **Not** green-key instability — `PROGRAMS` interning
(`bytecode.rs:1915-1926`) and the `DRIVERS` cache (`:1929-1940`) exist precisely
to keep `(program ptr, pc)` put across calls.

*Fix:* give the macro-generated portal the `ll_portal_runner` entry door.
`pyre-jit/src/eval.rs:9278` is the working in-tree pattern for this exact API —
follow it rather than inventing a parallel mechanism. Two things to get right:
the entry trace must run to `FINISH` and become a **procedure**, so it must not
auto-stamp a loop header (`pyjitpl.py:1546-1556`, `warmspot.py:766/775/790`
`no_loop_header`); and the back edge must not regress. Note `#[jit_interp]` is
**wasmi's** front-end too, so this is not a cel-only change.

*Gate, restated:* **P6–P8 are worth doing IF the procedure, once it can fire,
beats the interpreter per call.** The "it demonstrably is not firing, cause
unknown" evidence-against is now retired — the cause is known and is a missing
call site, not a property of straight-line traces. What remains genuinely open is
the economics: a straight-line procedure has no loop over which to amortize
entry/exit, so if the ~35–50 µs fixed per call is real it loses to a 7 ns walker
call by three orders of magnitude. **Measure that number after the door exists.
If it says no: stop at P5**, bank the data-model win, and accept
single-activation CEL as untraced.

### The economics half — measured 2026-08-07, and it says STOP AT P5

Filed as task **#88**. Two-point decomposition over the size ladders (`n=10`,
`n=10000`), **both backends**:

| ladder | majit fixed | majit /elem | clean fixed | clean /elem | break-even n |
|---|---|---|---|---|---|
| dynasm / map | 51 537 ns | 1.53 ns | 152 ns | 15.03 ns | 3 807 |
| dynasm / filter | 91 543 ns | 2.76 ns | 188 ns | 21.12 ns | 4 976 |
| cranelift / map | 47 202 ns | −0.32 ns | 115 ns | 9.01 ns | 5 047 |
| cranelift / filter | 34 494 ns | 4.55 ns | 119 ns | 14.42 ns | 3 483 |

The code generator **works** — 5–10× faster per element. It is buried under a
fixed per-call cost three orders of magnitude above a whole CEL evaluation.
`map_list_scaling/10`: clean 302 ns vs majit 51 552 ns = **170× slower with a
compiled loop present.** Prime suspect, labelled a hypothesis and not RCA:
**`gfails/call == 1.00` on all 12 compiled cases, both backends** — one guard
failure per call, forever.

⛔⛔ **Gate trap this establishes: `loops_compiled` reads 1 here.** The artifact
exists, runs, and is 170× slower than the tree-walker. Never report a compile
count without `gfails/call` and a ns/call number beside it.

**Verdict: stop at P5 for the JIT half.** But be precise about *why*, because
the tempting argument is unsound: the reasoning that the entry door only mints
procedures priced at this same 34–92 µs **transfers a number measured on LOOP
artifacts onto PROCEDURE artifacts that do not exist yet** — a different class,
reached by a different path (`ResumeFromInterpDescr::compile_and_attach` →
`attach_procedure_to_interp`, not `compile_loop`). The sound argument is the
structural one: a 5-op straight-line CEL predicate costing 43–130 ns in the clean
VM has almost nothing for a procedure to win back, and upstream's entry door pays
for functions with loops or long bodies, not 5-op expressions.

⇒ **#83 is still worth fixing, on its own merits, independent of this gate:** it
is a PyPy parity gap in majit (one call site), and `#[jit_interp]` is **wasmi's**
front-end too, where portal economics are nothing like a CEL predicate. Building
it also converts the transfer above into a measurement.

**Re-entry criterion for P6–P8:** a compiled cel artifact's fixed per-call cost
under ~1 µs on both backends. Until then the JIT half cannot pay for any CEL
program that is not iterating thousands of elements — which, per the census
below, is almost none of them.

**Census, for the record** (classification: a comprehension macro in receiver
position — `Expr::Comprehension` is the AST's only looping node,
`cel/src/common/ast/mod.rs:22`, `:146`):

| corpus | distinct | back-edge | frac |
|---|---|---|---|
| cometkim, by case | 29 | 18 | 62% |
| cometkim, distinct expressions (size ladders collapsed) | 18 | 7 | 39% |
| `parity_sweep_binary_operators` (+unary/ternary) | 5092 | **0** | 0% |
| all hand-written, full features | 725 | 86 | **11.9%** |
| `example/` — the user-facing programs | 8 | **0** | 0% |
| README "what CEL is for" snippets | 3 | 1 | 33% |

The corpora **over-represent** comprehensions: cel-jit's own JIT suite samples
them deliberately (they are the only thing that traces today), and spec
conformance makes them dense per *feature* rather than per *program*. The two
artefacts written to demonstrate the library are 1/3 and 0/2.

### P1 — cel hygiene + BUILD the differential oracle. Landable and revertible; no JIT.

**The oracle the plan leans on at P2/P3/P5 does not exist yet.**
`parity_sweep_binary_operators` (`cel/src/majit/mod.rs:5205-5227`) lives inside the
`#[cfg(test)]` module of the front-end-A columnar tier that P9 deletes; `sweep_case`
(`:4313-4330`) calls `lower_typed(...)`/`sum_reducible()` first and returns
`SweepVerdict::Declined` on failure, so **every expression the batch lowering
refuses is never compared**; the only assertion is `census[Agreed] > 200`
(`:5222-5226`); and its operand universe is a batch schema — 12 columns + 5
literals of int/uint/float/bool/string/timestamp (`:4249-4291`), with no maps,
lists, null, comprehensions/macros, errors, duration, or opaque.

So P1 **builds** it: a checked-in corpus file of CEL source → expected
`Value`/error, run against every registered evaluator, with a **per-CEL-feature
coverage assertion**. ⛔ "both evaluators" was wrong — a JIT-free build has
**one** (`pub mod majit` is `#[cfg(feature = "jit")]`). The oracle's `EVALUATORS`
table is built so registering the second is a one-row change; P2 adds
`resolve_value` there and the corpus becomes a three-way check with no other
edit. Freeze it as **data**, not as a live second evaluator — otherwise P5
rewrites the oracle and the subject in the same commit (the `#[cfg(test)]` walker
is itself written in tuple-variant matches).

Then the hygiene, each independently revertible:

- Replace `fn bool<'a>(bool) -> Cow<'a, dyn Val>` (`objects.rs:2161-2163`,
  `Cow::Owned(Box::new(CelBool::from(b)))`) with `static TRUE_VAL/FALSE_VAL`
  returned `Cow::Borrowed` — 13 in-eval call sites (`objects.rs:1636,1642,1812,
  1828,1830,1836,1852,1854,1861,1877,1888,1968,1978,1980`); results propagate out
  as `Ok(bool(..))` with no `.into_owned()` between, so the borrow survives.
- `LiteralValue::Null => Cow::Owned(Box::new(CelNull))` (`common/ast/mod.rs:62`)
  → `static NULL` + `Cow::Borrowed`; the lone allocating arm among seven.
- Delete `pub trait Lister` (`common/traits.rs:113-115`) — one occurrence in the
  workspace, its own definition.
- Delete `Value::Function(Arc<String>, Option<Box<Value>>)` (`objects.rs:1050`) —
  no producer; only `Debug :1074`, `type_of :1139`, `PartialEq :1193` reference
  it; the enum's only recursive `Box<Value>`.
- ⛔ **PREMISE FALSE — do not "unify" the two `Key` enums.** Verified against the
  source 2026-08-07: `cel::objects::Key` (`objects.rs:355-360`) is
  `Int(i64) | Uint(u64) | Bool(bool) | String(Arc<String>)` — **primitives**;
  `cel::common::types::map::Key` (`map.rs:165-170`) is
  `Bool(CelBool) | Int(CelInt) | String(CelString) | UInt(CelUInt)` — **`dyn Val`
  wrapper types**. Different universes, and `map::Key` is the one **P2 deletes**
  along with the rest of the `dyn Val` world. Unifying them would merge a type
  into one that is about to be removed.
  ⛔ `RecordSchema::position` "makes a virtual call per field" is **also false** —
  `keys` is `Vec<Key>` (`objects.rs:68`), a concrete type, so the scan is static
  dispatch; the one virtual call is already hoisted.
  ✅ The **sound half** was landed: the genuinely duplicate `KeyRef`/`AsKeyRef`
  and their trait-object impls collapsed onto the `objects` copy.
  ⇒ **#85's "one genuine duplicate leaf" framing does not survive**; re-derive
  #85 against P2's deletions rather than against this bullet.

**Verification:** a committed allocations/eval number (a direction is not a gate);
in-tree tests green; `cargo bench --bench runtime` within a committed tolerance.

⛔ **Two counts in this document were JIT-ON counts.** "241 in-tree tests" is a
`--features jit-*` number — 123 of the 232 `#[test]`s under `cel/src` live in the
jit-only `majit` module; **JIT-free is 103**. Likewise `allocs_per_eval` reports
**64 rows only with `--features jit-cranelift`/`jit-dynasm`**; JIT-free is **52**,
and the 12 `regvm/*` rows print as "in the baseline but NOT measured in this
configuration". Quote the configuration with any of these numbers.

### P1 — LANDED 2026-08-07, `c2808ab..68e433b` (8 commits on cel-jit `majit`)

**The oracle exists**: `cel/tests/oracle.rs` + `cel/tests/oracle_corpus.txt`,
**156 cases**, frozen as **data** — the harness renders a `Value` into the corpus
notation and string-compares; nothing parses the `want:` side back into a value.
`EVALUATORS` holds one door today and P2 adds `resolve_value` as a second row with
no other change. Coverage assertion = **45 axis points** with per-point floors plus
unknown-tag rejection, and it is **proven non-vacuous** (deleting the 4
`macro_filter` cases fails, naming both `macro_filter` and `comprehension_nested`).
Not covered: the JIT tier, `Value::Struct`, opaques beyond `OptionalValue`, error
payload detail, parse-error detail.

**Allocations (the load-independent gate), verified independently:**
`rows=52 unstable=0 drifted=24 off-thread-contaminated=0`, and **all 24 drifts are
negative — zero regressions**. `filter_list_scaling/10000` 40022→30022,
`comprehension_scaling/500` 2782→2282, `walker/comparison` 4→**0**. No timings
taken (the box was loaded); `cargo bench` deliberately skipped. The baseline was
**not** re-blessed, because it was blessed with JIT on and re-blessing JIT-free
would drop its 12 `regvm` rows.

⭐ **Hygiene item 1's site list in this document was incomplete**: six more
hand-inlined `CelBool` boxes live in the `LOGICAL_OR`/`LOGICAL_AND` arms and never
went through `fn bool`, plus one in `string.rs` `contains`. (`struct.rs:127/133/138`
look like more but are inside `#[cfg(test)]`.)

**Two real defects found by the oracle** — `optional == optional` is always false
including against itself (`Optional` never overrides `Val::equals`; the default at
`common/value.rs:64-66` returns `false`), and `type()` is unimplemented (nothing
registers it; `dyn()` beside it is registered at `common/types/dyn.rs:16`). Filed
as tasks.

⛔ **`KNOWN` is a comment convention, not a harness feature.** `rg 'KNOWN'
cel/tests/oracle.rs` returns **nothing**; the two markers are `#` comment lines in
the corpus (`:433`, `:600`). What actually gates is the literal `want:` line — so a
"known bug" is **asserted as the wrong answer**, with a comment saying why. There
is no marker to flip later.

⇒ ⭐⭐ **A defect pinned this way must be fixed BEFORE a second evaluator is
registered, not after.** The corpus is a *shared* expectation across every row of
`EVALUATORS`; the moment two evaluators disagree, no single `want:` can satisfy
both and the oracle goes red at exactly the step designated as the gate. This is a
general property of the design, not a quirk of one bug — see P2.

**Two stacked pre-existing reds cleared** so `cargo test -p cel` builds JIT-free at
all: `livescale.rs`/`poison.rs` were on disk but absent from `[[example]]`, so they
lacked `required-features = ["jit"]`; clearing that exposed a stale doctest
(`magic.rs:215`) that had gone bad when `ListRef::iter` started yielding owned
`Value` (`objects.rs:931`) under #80/#81. The outer red had been masking the inner
one. `cargo test -p cel` is now green: 103 lib + 4 oracle + 15 doctests, 0 failed.

### P2 — delete `dyn Val`; `Value` becomes the walker's single universe.

Three commits, not one: (a) add `resolve_value(expr, ctx) -> Result<Value,_>`
alongside `resolve_val`, arm by arm, gated by the **P1 oracle**; (b) flip
`Program::execute`; (c) delete `resolve_val` and the old universe.

Deletions: `trait Val` and its twelve `as_*`/`into_*` accessors
(`common/value.rs:9-69`); the twelve capability traits (`common/traits.rs`);
`impl ToOwned for dyn Val` (`value.rs:94-100`, which silently makes all 17
`.into_owned()` sites in `resolve_val` allocate); `dyn Val::downcast_ref`
(`value.rs:71-75`); **both** bridges — `TryFrom<&dyn Val> for Value`
(`objects.rs:1367`) and `TryFrom<Value> for Box<dyn Val>` (`objects.rs:1456`).

Also required in the same change:
- `Iterator<'a>::next(&mut self) -> Option<&'a dyn Val>` (`traits.rs:79-81`) →
  `-> Option<Value>`. The borrowed-element return is *why* a columnar or record
  strategy is unimplementable and why `Value::List` must be exploded at
  `objects.rs:1471` before iteration.
- `Context::{Root,Child}.variables` (`context.rs:38, :44`) →
  `BTreeMap<Box<str>, Value>`; `add_variable`/`add_variable_from_value`
  (`:78, :88`) and the resolver read path (`:153, :168`) stop calling
  `try_into().unwrap()`.
- Replace `add_variable_as_val(name, Box<dyn Val>)` (`context.rs:91-127`, a
  documented 27-line extension point) with a lazy-object registration so the
  advertised protobuf/DB-row story survives.
  ⛔ **Not `W_OpaqueObject`** — that is a §3 class-family type that does not exist
  until **P5**, so the original instruction was unimplementable here. Use
  `Value::Opaque(Arc<dyn Opaque>)` (`objects.rs:1061`), which is already in the
  tree and is already the encoding for optionals — there is no `Value::Optional`
  variant; `Value::Opaque(Arc<OptionalValue>)` is it (`objects.rs:764`). That
  preserves the extension story without minting a P5 type early.

⭐ **Two findings that shrink (a) substantially** (verified 2026-08-07):
`Value` **already has a full native operator set** — `impl ops::Add/Sub/Div/Mul/Rem
<Value> for Value` (`objects.rs:2159, :2203, :2239, :2269, :2294`) armed across
Int/UInt/Float/List/String/Duration/Timestamp, with error spellings the P1 corpus
already pins, plus `PartialEq`/`Eq`/`PartialOrd`. So `resolve_value` is **built on
these, not written from scratch**. And `Env::functions` is empty by default —
nothing in-tree populates it, so `find_overload` always misses in the corpus, which
makes `common::functions::Function`'s signature change cheap to verify but a real
public break to announce.

⭐⭐ **Expect a class of divergences, not one.** Any place the two universes answer
differently forces the same "fix before registering row 2" ordering as #93 did. For
each one, adjudicate **which answer the CEL spec requires** — do not reflexively
make `dyn Val` match `Value`. #93 went that direction only because `Value` was the
correct side (`PartialEq for Value` routes `Value::Opaque` through `opaque_eq`,
`objects.rs:1213`, and `OptionalValue` derives `PartialEq`/`Eq`, `:726`).

**Verification:** the P1 oracle is the gate and must run at (b) before the walker
is deleted. Allocations/eval: `TryFrom<Value> for Box<dyn Val>`'s list arm
(`objects.rs:1471-1475`) costs 2N + Vec + Box per bound list, so a list-binding
case should drop by ~2N — commit the number. **This closes #82 by deletion and
renders #85 moot**: `struct_leaf_counts` (`majit-translate/src/lib.rs:1482-1488`)
counts `rsplit_once("::")` leaves and `front/mir.rs:1641, :1678` insert
`{enum}::{Variant}` rows, so the collisions were `Value::Bool` vs the struct
`Bool`; with the trait families gone there is nothing left to skip.

### P2 — LANDED 2026-08-07, `988a100..9217012` (9 commits on cel-jit `majit`)

`dyn Val` is gone. Landed in two series (6 then 3) on top of P1's `68e433b`;
final tree `d8a86f1e2f3b…`. Verified independently of the report: default
**87 lib + 4 oracle + 15 doctests**, `structs,json,bytes` **101 + 4 + 17**,
`cargo check -p cel --all-features --all-targets` clean with **0 warnings**.
Only two `dyn Val` strings survive tree-wide, both doc prose describing the
deleted walker in past tense.

**#82 closed by deletion, as designed.** `bind/list-to-context/1000`
**1002.000 → 0.000**, and zero at every size (1/10/100/1000). Scaling rows went
with it: `filter_list_scaling/10000` 40022→18, `map_list_scaling/10000`
30023→19, `comprehension_scaling/500` 2782→26, `member_access` 10→0.
52 rows, 0 unstable. A 2200x fall is also the shape of a benchmark that stopped
doing its work, so it was checked rather than reported: filter over 10000
returns 5000 elements, first 0 last 9998; `map().size()` returns 10000.

⭐ **Attribution inside P2 was measured, not assumed.** Instrumenting at three
points in identical config showed the **boundary flip** (`0002`) moved the five
host-overload rows — `string_operations` 20→3, `custom_function` 6→2,
`real_world_policy` 13→4, `comprehension/bind+exec/size/100` 116→13 — and the
**deletion** (`0003`) moved **zero**. A pure deletion being allocation-neutral
is the expected result; reporting it that way avoided crediting the win to the
wrong commit.

#### ⛔ Where this plan was wrong

- **"Three commits, not one" understated it — it was nine**, and the ordering
  constraint was real: #93 (`Optional::equals`) had to land *before* (a), not
  "later in its own commit". `KNOWN` in the corpus is a **comment convention
  with no harness code behind it** — the `want:` line is the assertion — so a
  pinned defect must be fixed before a second evaluator registers, or the
  oracle goes red at the step designated as the gate.
- **`Value::Struct` blocks the last of the deletion and this plan never says
  so.** `Value::Struct` holds `Arc<CelStruct>`, and `CelStruct` is itself a
  `dyn Val` type whose `field_values()` returns `BTreeMap<String, Arc<dyn Val>>`
  (`common/types/struct.rs`). Finishing required redesigning `Value::Struct` to
  hold `Value` fields — real work, not a rename. Behind non-default `structs`.
- **The `Opaque` substitution preserves the extension point but NOT the
  advertised capability.** `Value::Opaque(Arc<dyn Opaque>)` carries equality,
  debug and a type name and **no accessors**, so a value bound that way cannot
  be indexed, iterated or sized. The protobuf/DB-row story is gone until the
  value family grows accessors — deferred to P5 rather than grown here, and
  stated in the doc comment instead of implied away.
- **"`Env::functions` is empty by default" is wrong as a gating claim.**
  `Context::default()` → `Env::shared_stdlib()` registers every overload, and
  `find_overload` runs *before* the `FunctionRegistry`. The oracle did gate the
  overload-matcher flip normally.
- **"renders #85 moot" is not yet established.** The trait families are indeed
  gone, but #85 needs re-deriving against what front-end B actually sees now —
  one `Value` enum, no trait objects — not closing on this plan's prediction.

#### Behaviour changes, all pinned in the corpus

Corpus 156 → 178 records. `[1] + {"a": 1}` was `list[int(1), string("a")]`
(it concatenated a map's *keys*) and is now `NoSuchOverload`;
`duration + timestamp` was refused and is now the sum. Three reachable library
panics became a typed error — `(1).value()`, `(1).or(…)`, `(1).orValue(2)` now
return the `UnexpectedType{got:"int", want:"optional_type"}` that `hasValue`
always gave, because `OPTIONAL_TYPE` is `Kind::Opaque` over `DYN_TYPE`, making
`is_assignable` unconditionally true so those overloads were selected on name
and arity alone. `m.map(k, k)` is deliberately **not** pinned: the two
evaluators answered `[b, a]` and `[a, b]`, so map-range order is genuinely
unspecified and only order-independent results are frozen.

16 unit tests were dropped, accounted for by exact name diff: ten tested a
deleted subject with no surviving behaviour; the six asserting still-observable
answers became corpus cases.

⭐ **Method note that changed the answer twice.** Deriving the operator error
matrix by *reading* the impls gave the wrong answer — a fixed grep window bleeds
into the neighbouring `impl` and misattributes its terminal error. Sweeping all
1694 (receiver, operator, argument) triples through both evaluators and diffing
is what settled it, and the same sweep then found 15 further divergences,
including four in `PartialOrd for Value` and a missing `Bytes` arm in `Add`.

⚠ **Open**: `allocs_per_eval.baseline` was blessed under
`profile=dev features=regex,chrono,jit` and prints a NOT-comparable banner
against every JIT-free run, so it currently gates nothing anyone runs.
Re-bless under `jit-cranelift`, stating in the message that it blesses an
*improvement* (with before/after rows) and that the JIT config yields 64 rows
vs 52 — and if one config leaves `cargo test -p cel` ungated, keep two
baselines. See also #97: `--no-default-features --features regex` does not
compile at all (`ser.rs:980`, ungated `use chrono::FixedOffset`), pre-existing.

### P3 — the code object and a bytecode VM over `Value`.

`Program` gains `Arc<CelCode>` and the §6.4 `bind` step. Compile the AST to a flat
bytecode covering **all** of CEL, resolving every variable and builtin to a
code-object index at compile time. Make the compiler **exhaustive-match** on
`cel/src/common/ast/mod.rs`'s `Expression` variants so a new AST node is a compile
error, not a runtime fallback; enumerate the opcode set against `has`/`all`/
`exists`/`exists_one`/`map`/`filter`, `type()`, `dyn()`, optional syntax, `in`, and
field selection on Struct/Map/Opaque.

Activation record = one fixed-size array sized at compile time; nested
comprehensions get distinct compile-time slot ranges in the **same** record — no
frame stack; CEL has no functions of its own. The comprehension's
`while let Some(item) = items.next()` (`objects.rs:2062-2118`) becomes an explicit
back edge; `ComprehensionExpr` already carries iter_range/iter_var/accu_var/
accu_init/loop_cond/loop_step/result (`common/ast/mod.rs:146-155`). Build the
frame as the `W_ObjArray`-backed shape from §5 now, even though it is not yet
virtualizable.

**Acceptance criterion, not a note: no loop anywhere reachable from the dispatch
loop except the dispatch loop itself.** `policy.py:48-68` forces such a graph
residual. "The only back edge in CEL is a comprehension" is a *consequence of
enforcing this*, not a property of the language — `in` over a list, map-key
lookup, `contains`/`startsWith`/`matches`, and list/map equality all loop. One
inlined looping helper introduces a second back-edge class with no code-object
green. Make it a test, not a convention.

⚠ Front-end A **unrolls literal-list comprehensions** (green length,
`cel/src/majit/lower.rs:2690-2691`); only a runtime-list comprehension emits a
real inner back edge (`:3740-3766`). So `[1,2,3,4,5].map(x, x*2)` and its
siblings are comprehensions with *no* back edge today — likely part of #83's
18-vs-12 gap. Under P3 every comprehension becomes an explicit back edge, so
classifications taken against front-end A do not transfer. Errors out-of-band per §6.3 from the first commit — retrofitting
`?` out later is a rewrite.

Keep the tree-walker behind `#[cfg(test)]`; keep the front-end-A columnar tier
(`majit/bytecode.rs`, `lower.rs`, `batch.rs`) untouched.

**Verification:** 241 tests + the P1 oracle green with the VM as
`Program::execute`. Committed allocations/eval and ns/eval on
`majit_vs_cometkim_percall`.

### P4 — cel repo integration (STARTED at P0; it is real work, not config).

cel is a separate git repo, pins majit by rev, depends only on
majit-ir/majit-macros/majit-metainterp [verified], and has **no `build.rs`**.
Front-end B's harness is not a reusable crate: it is `pyre/pyre-jit-trace/build.rs`
(1355 lines) with a `MAJIT_LLBC_EXTRACTION` bootstrap for the self-extraction cycle
(`build.rs:44-48, :72-80`), a preflight, a versioned codegen cache, a 1 GiB worker
stack and 13 generated artifacts (`build.rs:19-34`), plus `scripts/llbc_extract.py`.
pyre escapes the cycle because `pyre-jit-trace` extracts *sibling* crates.

So: split cel into `cel-runtime` (value universe + VM) and a `cel-jit-trace` build
crate that Charon-extracts it; add `majit-translate` and `majit-gc` as direct deps;
port the `CrateSpec` extract driver (`pyre/scripts/extract-llbc.py:26-83`).

**Verification:** the pipeline runs on cel in CI and emits the P0.b census as a
checked-in artefact.

### P5 — the class re-lay (SEMVER MAJOR). Depends on M1. **Must leave a shippable, JIT-free cel.**

Land §3, §4, §7. `Value` becomes `#[repr(transparent)] struct Value(CelRef)` with
constructors and `kind()`; tuple-variant `match` disappears. Decide the
`W_OpaqueObject` reclamation contract (D12) and the rooted-handle API here — not
at P6.

Breaks that must be planned, not discovered: `const V: Value = Value::Bool(false)`
(`benches/runtime.rs:48-49`); direct variant construction in
`fuzz/fuzz_targets/value_binop.rs:71-72`; the rustdoc at `objects.rs:648-650`;
`magic.rs:8-17`'s `impl_conversions!` binding `Arc<String>` / `Arc<Vec<u8>>` /
`ListRef` / `Arc<dyn Opaque>` into third-party signatures (documented at
`magic.rs:117`, used in-tree at `functions.rs:158, :280-281, :316`);
`Identifier(pub Arc<String>)` (`magic.rs:190`); the `Send`/`Sync` removals in §7;
and `chrono` (`cel/Cargo.toml:16`) — every timestamp builtin becomes a call over a
changed representation. Ship `From<Arc<String>>`-shaped compat impls where
possible. `ExecutionError` keeps `Clone + PartialEq` (`lib.rs:68`) — a *derived*
`PartialEq` on `Value(CelRef)` would be pointer identity, so **write it by hand**
to call the structural comparison (the doctest at `objects.rs:648-650` asserts
`Value::Opaque(Arc::new(MyId(7))) == Value::Opaque(Arc::new(MyId(7)))`).

**Verification:** 241 tests + the P1 oracle green. Allocations/eval from **both**
counters. ns/eval. **This is the phase to stop at if P6's census disappoints or
P7 turns out to be the 59k-line lift** — everything after is JIT-only, so P5 must
ship on its own.

### P6 — GC + descr registration; re-census; go/no-go on the JIT half.

Three registrations, none optional, none reconciled by the compiler:

- **(a)** `MiniMarkGC::with_config` → `register_type(TypeInfo::object_subclass(..))`
  once per class in a fixed order → `freeze_types()` before any compiled code runs
  (`subclassrange_{min,max}` are assigned only there, `majit-gc/src/trace.rs:872-883`).
- **(b)** the descr side with the **same numeric tid**, via
  `make_simple_descr_group_keyed_with_headerless(index, size_of::<W_IntObject>(),
  tid, path_hash(def_path), &CEL_INT_CLASS as *const _ as usize, …)`
  (`majit-ir/src/descr.rs:4963-5029`), with an equality assert at init exactly as
  pyre does (`pyre-jit/src/eval.rs:1376`). The two tid namespaces are independent
  (`GcCache::alloc_type_id`, `descr.rs:899-906`, vs `MiniMarkGC::register_type`'s
  `TypeRegistry`, `trace.rs:810-846`) and a mismatch is a **silent mis-trace**: the
  JIT stamps `descr.type_id()` into the header (`majit-gc/src/rewrite.rs:1153-1163`)
  and the collector reads that word back to index its own registry
  (`collector.rs:5360-5363`). `cache_key` must be `path_hash(def_path)` matching
  what the translator hashes, dual-published under the simple name as pyre does
  (`pyre-jit-trace/src/descr.rs:706-798`).
- **(c)** `JitDriver::set_gc_allocator(Box::new(..))`
  (`majit-metainterp/src/jitdriver.rs:1814-1816`) plus `set_new_via_gc(true)` on
  dynasm (`majit-backend-dynasm/src/runner.rs:1815-1827`), or compiled `New`
  allocates with `libc::malloc` and never enters the traced heap. cel has **zero**
  such call sites today.

Do **not** use `#[jit_struct]`: its tid comes from `GcCache::alloc_type_id` and it
registers no collector trace shape (`majit-macros/src/jit_struct.rs:122-135`; its
only user is a smoke test).

Then land M3 and emit `record_exact_class` after every VM-internal narrowing (§4's
taken arm, a strategy read, a `to_key`). Upstream drops the op and installs only
`_known_class` (`rewrite.py:386-396`), so it is free in the compiled trace.

Land M5 and give `W_ListObject` its real strategies — unboxed int/double column,
record window, generic object array. Array-virtual constraints: a `NEW_ARRAY`
virtual needs a compile-time-constant length in [0, 150000] and **constant indices
on every access** (`virtualize.py:214-221` + `info.py:489-492`; `:276-308`); one
variable index forces the whole array. So the strategy serves the variable-index
case from a real object and only the small-literal case as a virtual.

**Verification.** Census: `new_ops` at every value-construction site, each
`NewWithVtable` carrying a **nonzero vtable**; residual:inline better than the
**measured** walker baseline; `record_exact_class` present.

⛔ **This gate used to read "better than 8:1", and 8:1 is stale.** It came from
§1's walker cell `577 : 73`. Re-derived 2026-08-08 (#107), the walker measures
**656 : 145 = 4.5:1** — so a P5 census landing anywhere between 4.5:1 and 8:1
would have **passed a gate while being no better than the baseline it is
supposed to beat**, and the GO/NO-GO below would have read as a win. Restate the
bar against a baseline re-measured on the same tree as the candidate, not
against a number transcribed from this table. ⚠ The absolute is tree-conditioned
(#107); the point stands under any of them, since the gate needs a *contemporary*
comparison, not a historical constant. Runtime: the non-pyre setup recipe already exercised
in-tree at `majit-backend-dynasm/src/runner.rs:4209-4259` —
`supports_guard_gc_type` / `check_is_object` / `get_actual_typeid` answer correctly
for cel's classes with no pyre in the process. Barrier audit test (§7).
Allocations/eval from the in-heap counter.
**GO/NO-GO:** if this census is not decisively better than P0.b's, stop. P5 banked
the data-model win.

### P7 — close the #84 gap list from P0.c.

Whatever P0.c says is missing, extracted into majit-metainterp behind the
jitcode/descr interfaces the blackhole already uses (`executor.rs` already consumes
`majit_translate::jitcode::BhDescr`). Scope it by cel's actual opcode surface from
the P6 census — cel needs no vable-across-frames ops, no cross-frame
exception-handler scan, no `direct_assembler_call`. Its own PR series, its own
tests; cel does not block on it (P1–P6 ship without it).

**Verification:** the lifted tracer reproduces **P0.c's golden file** op-for-op
before any cel use.

### P8 — merge point, virtualizable, first compiled trace.

Land M2 and M8. Declare `virtualizables: ["frame"]` per §5–§6 with
`locals_stack_w[*]` and the depth promoted right after the merge point. Promote
registered host functions to a constant at the call site (`jit.promote(self.code)`
shape, `pypy/interpreter/function.py:91-96`). `matches` either interns its compiled
`Regex` — today `regex::Regex::new(&regex)` runs on **every** evaluation
(`cel/src/functions.rs:277-286`), so there is no stable pointer to key an
rsre-style sub-driver on — or stays a plain residual call; **never compile a regex
inside the traced loop.** Classify every opaque leaf deliberately
(`rpython/jit/codewriter/call.py:281-315`): elidable / loop_invariant / plain
dont_look_inside / not_in_trace.

**Verification:** a trace forms, closes, and compiles. **Allocations per iteration
on a comprehension is the number that matters — target zero.** The P1 oracle green
with the JIT on. Per-shape guard-failure census
(`cel/tests/majit_trace_evidence.rs` already pins this shape). Force-point census:
name which emitted op forced which virtual. The two triggers are
`Optimizer::_emit_operation`'s unconditional arg loop (`optimizer.py:650-652` /
`optimizer.rs:4705-4720`) and OptEarlyForce, which exempts only `SETFIELD_GC` /
`SETARRAYITEM_GC` / `SETARRAYITEM_RAW` / `QUASIIMMUT_FIELD` / `SAME_AS_*` /
`OS_RAW_FREE` (`earlyforce.py:15-29`); guards do **not** force
(`resume.py:210-226`), and a store into a real object defers into the guard's
`pendingfields` until the next non-exempt side-effecting op (`heap.py:610-639`).

### P9 — retire the mirror; land the #83 decision from P0.f; dissolve #53/#54.

Demote or delete the front-end-A columnar tier. Execute the #83 decision **taken
at P0.f** — which is now "does the compiled procedure fire and pay", not "where
do we find a back edge".

⚠ **The old escape hatch was mis-priced and rsre was cited for a mechanism it
does not have.** Two corrections [verified 2026-08-07]:

* `find_jit_merge_points` (`warmspot.py:175-181`) asserts
  `len(seen) == len(results)` over **graphs**, i.e. *"found several
  jit_merge_points in the same graph"* — stricter than "same portal". majit
  enforces it independently: `register_configured_jitdrivers`
  (`majit-translate/src/lib.rs:1921-1959`) rejects two specs resolving to one
  portal graph. `PipelineConfig.jit_drivers` is a `Vec`, so N drivers are fine —
  each just needs its own free function. **That part is cheap.**
* **But the loop the hatch would attach to is scheduled for deletion.** cel's
  only per-row loop is emitted by front-end A's lowerer
  (`cel/src/majit/lower.rs:879`, backward jump at `:1122-1128`) inside
  `#[cfg(feature = "jit")] pub mod majit` — not stable API, and P9 deletes it.
  §6.1's portal is one activation of one expression. **Post-P3 there is no row
  loop to move a merge point to**, so the hatch first requires inventing a
  public batch entry (`BoundProgram::execute_many`). Real cost: new public API +
  second portal + M2, not "one `JitDriverSpec`".
* rsre's actual answer is **eleven** drivers, each in its own function
  (`rsre_core.py:383, 414, 482, 549, 611, 1225, 1382, 1413, 1431, 1453, 1471`),
  each at the top of whatever real loop that construct has; the four *search*
  drivers loop over the **input string position**, not the pattern. The cel
  analogue of that is "a loop over inputs", which one `Program::execute` does
  not have.

#53 (TLS program/driver caches) and #54 (a refused batch re-running every row in
the other evaluator) dissolve mechanically once there is one interpreter and one
code object.

---

## 10. Where the filed tasks land

| task | resolution | phase |
|---|---|---|
| **#82** — binding a `Value::List` boxes every element | dissolved by deleting `TryFrom<Value> for Box<dyn Val>` (`objects.rs:1456`, list arm `:1471-1475`, 2N allocs); the `W_ListObject` strategy is the structural form | P2 (deletion), P6 (strategy) |
| **#85** — leaf-keyed trait-family skip | **moot for cel**: P1 removes the one genuine duplicate leaf (`Key`), P2 deletes all twelve `dyn Val` families so there is nothing to auto-register. The config bypass (`register_trait_families`, `lib.rs:1529-1554`, skipping the cross-registry bail at `:1584-1595`) is documented but not needed | P1 + P2 |
| **#55** — virtualizable frame slots | **mandatory prerequisite, not optional** — §2(b), §5 | P3 (shape), P8 (declaration) |
| **#84** — no interpreter-neutral runtime tracer | re-scoped at P0.c against the BC-keyed `JitCodeMachine` evidence; golden file captured there; residual gap closed at P7 | P0.c → P7 |
| **#83** — `COMPILES == 0` on single-activation CEL | ⛔ **re-framed**: not "no back edge, cannot trace" but "the FINISH-from-`ResumeFromInterpDescr` procedure path exists, is ported, and is *not firing*" — a behaviour defect to diagnose. #83's own counter would have caught a procedure. Decided by the P0.f experiment | P0.f → P9 |
| **#53 / #54** | dissolve once there is one interpreter and one code object | P9 |

---

## 11. Named deviations from upstream

**D1 — Rust has no inheritance.** `#[repr(C)]` first-field embedding stands in for
RPython's `super` substructure. Smallest possible deviation: `rclass.py:140-158`
documents the RPython layout as literally `struct X { struct Y super; … }`. pyre
already relies on it (`pyre-macros/src/lib.rs:1306-1325`).

**D2 — Rust has no `import_from_mixin`.** The twelve capability method-groups
become free functions selected by the narrowing chain, not copied into a class
body. Upstream copies functions too (`func_with_new_name`,
`objectmodel.py:1104-1157`) precisely because multiple inheritance is forbidden
(`classdesc.py:549-550`), so the semantics match.

**D3 — no per-class function-pointer vtable.** Upstream installs
`shortcut___add__` as a slot on the class (`typedef.py:240-269`). We use an
explicit narrowing chain, because front-end B cannot lower a call through a
fn-typed struct field (§2a). Same observable devirtualization, different spelling.
Revisit if `operand_is_fn_ptr` is ever generalized.

**D4 — `ListStrategy` is an inline `#[repr(u8)]` tag, not a prebuilt strategy
object.** Upstream's `W_ListObject.strategy` is a pointer to a singleton whose
method call is a typeptr read (`listobject.py:294-296`). Ours is a discriminant
read plus a narrowing chain — same guard-then-fold, one fewer object, and it avoids
putting a `CelRef` inside a prebuilt (which would disqualify the immortal
allocation path, `majit-gc/src/header.rs:188-196`). Reversible.

**D5 — one header word, not pyre's two.** `PyObject{ob_type, w_class}`
(`pyre-object/src/pyobject.rs:65-69`) needs `w_class` for user-defined Python
classes; CEL's universe is closed except Struct/Opaque, which carry their CEL
`Type` as an ordinary subclass field. This is *closer* to upstream, whose OBJECT
has exactly one word.

**D6 — `set_vtable_offset(Some(0))` while upstream 64-bit defaults
`gcremovetypeptr = True`** (`rpython/config/translationoption.py:94-95`). We keep
the word because `fuse_boxing_alloc` resolves the vtable from an `ob_type` store
and declines on zero (`model.rs:2998-3011`), so the no-word shape is unreachable
through front-end B's allocation lowering. Cost: one word per value.

**D7 — non-moving old-gen, not `incminimark`'s moving nursery.** Upstream defaults
`gc = 'incminimark'` (`translationoption.py:17`) and PyPy's target refuses anything
else (`targetpypystandalone.py:352-354`). We defer because cel has no shadowstack
pass. This is pyre's own stepping stone with pyre's own stated reason
(`gc_interp.rs:11-16`) and pyre's open #66.

**D8 — no GIL, and therefore `Value: !Send`.** Upstream and pyre are both
GIL-protected; `majit-gc/src/lib.rs:1002-1009` justifies `unsafe impl Send for
GcHandle` on exactly that. cel has no GIL, so we take the compile-time break
instead of the unsafe impl (§7). The single largest public-API deviation.

**D9 — `Value` keeps `Clone` and `PartialEq`, which `W_Root` does not**
(`__slots__ = ('__weakref__',)`, `baseobjspace.py:35-40`). Forced by
`ExecutionError` deriving both with eleven `Value`-carrying variants
(`lib.rs:68-137`). `PartialEq` is hand-written, never derived (§P5).

**D10 — the tree-walker survives as a `#[cfg(test)]` oracle through P2 only; from
P2 the oracle is a frozen data corpus.** Upstream has exactly one interpreter at
every point in its history. Freezing to data at P2 is what stops P5 from
co-mutating test and subject.

**D11 — no `guard_class` on the critical path.** Upstream gets it free from the
rtyper (`jtransform.py:1004-1010`). We get class facts from `NewWithVtable` and
`record_exact_class` instead (§4) because the guard_class vertical slice does not
exist in majit at any layer (M4). A sequencing deviation, not a shape one.

**D12 — `W_OpaqueObject` host-object lifetime.** `Arc<dyn Opaque>` freed the host
object for nothing; a side table does not. Either a finalizer or the contract
*"opaque host objects live as long as the heap"*. **Decide at P5 and record the
choice here.**

**D13 — errors are out-of-band, not `Result`, inside the traced VM** (§6.3). This
matches PyPy (`OperationError` on the ExecutionContext) and diverges from cel's
current Rust-idiomatic `Result` plumbing, which is reconstructed at the public
boundary.

---

## 12. Unmeasured, unverified, or open

1. **Every number in §1 is inherited.** No probe, build or benchmark was run
   producing this document. P0.b is the gate.
2. ~~The four P0.d lowering assertions are unexecuted.~~ **Three ran on
   2026-08-07 (P0.d above): 1 and 3 hold; 2 holds conditionally and pins M1 as
   the sole gate; 4 is not checkable below the codewriter and is still open.**
3. **`simplify.rs:79` removes `RecordExactClass` outright** in the OptSimplify pass
   while `rewrite.rs:2200-2233` installs the class fact. Which pass runs in which
   production pipeline configuration is unknown. If a cel trace takes the simple
   pipeline, **every M3 hint is silently dropped.**
4. **`try_const_fold_pure_field` (`majit-metainterp/src/trace_ctx.rs:1881`) has zero
   callers repo-wide**; the only live copy of `pyjitpl.py:876-880` is inside
   `pyre-jit-trace` (`state.rs:3376`, `jitcode_dispatch/heapcache_ops.rs:510`).
   Whether the pure-getfield-on-Const fold is reachable from the neutral tracer is
   a P0.c question.
5. **Back-edge shape stability is unproven for CEL.** `virtualstate.py:141-176`
   (field-descr identity at `:155-160`) raises `VirtualStatesCantMatch` on any
   divergence in known_class or fielddescr list, and `make_inputargs(force_boxes=True)`
   then materializes at the JUMP — one alloc per iteration. `[1, "a", true]` is
   legal CEL. `VStructStateInfo` matches on `typedescr is other.typedescr`
   (`virtualstate.py:223-231`). Measure at P8; a per-shape trace may be required.
6. **Cranelift and wasm `New` routing through the active GC** is claimed only by a
   doc comment inside the dynasm backend (`runner.rs:1817-1820`). Unread.
7. **No non-pyre majit-gc integration exists in this repository.** aheui is named
   only in a doc comment (`runner.rs:1822-1824`).
8. **`fuse_boxing_alloc` has no known configuration knob.** Only the hard-coded
   matcher was read.
9. **Out-of-repo consumers of cel's public API are invisible.** Only `cel/src`,
   `cel/examples`, `cel/tests`, `cel/benches`, `example/` and `fuzz/` were audited.
10. **Whether an existing bench regresses.** The typed register machine
    (`bytecode.rs float_bank`) has 4 Drop terminators and no header at all; the new
    class universe pays a header word and a pointer chase the register bank does
    not. P0.a's corpus **must** include the register machine's cases.
11. **Whether `LowerError::Unsupported` inside the portal's own graph kills the
    whole portal or only that callee.** The error site was read
    (`front/mir.rs:7996-8001`); its consumer was not.
12. **The "241 in-tree tests" and "4693 expressions" figures** were read from
    construction, not run.

---

## 13. Tripwires — silent failures with no error

Each needs a **CI assertion**, not a comment.

| failure | detection |
|---|---|
| `fuse_boxing_alloc` declines (wrong arity, wrong field name, zero vtable) — bare `continue` at `model.rs:2993, :3009-3011` | assert `new_ops` count == value-construction-site count, and that every `NewWithVtable` carries a nonzero vtable |
| descr tid ≠ collector tid — the JIT stamps one word and the collector reads it back | equality assert at init (`pyre-jit/src/eval.rs:1376` shape) |
| `offset_of(ob_header) != 0` on some leaf | `const { assert!(offset_of!(T, ob_header) == 0) }` in the class-registration macro |
| the interpreter is written as `match v.kind()` instead of `w_type(a) == &CEL_INT_CLASS` — no annotator narrowing, no typeptr read, `guard_class` stays 0 forever | census assertion on the emitted op shape for a canonical binop |
| a prebuilt is a plain Rust `static` — `guard_is_object` reads `[obj - 8]` off rodata | assert every prebuilt address came from the immortal allocator or the prebuilt-root registry |
| a `CelRef` field written without a write barrier | audit test: no `CelRef` field is assigned outside the `set_ref` helper |
| `set_new_via_gc(false)` on dynasm — compiled `New` goes to `libc::malloc`, object never enters the traced heap | runtime assertion in the `runner.rs:4209-4259` smoke-test shape |
| a `?` or `Result` reachable from the portal (§6.3) | grep/ast-grep audit test over the `vm::` module |
| an emitted op names a value inside the hot loop and forces it | per-trace force-point census, P8 |
| a stale majit git rev in `cel/Cargo.toml` | the rev-bump + re-census step attached to every M row (§8) |

### The twelve citations P0.e must re-verify (by symbol, not by line)

`model.rs fuse_boxing_alloc` args.len()==1 · `model.rs` ob_header→ob_type resolve ·
`model.rs` zero-vtable `continue` · `front/mir.rs` raw-deref→FieldRead ·
`front/mir.rs` the `"PyType"` root hardcode (both sites) ·
`codewriter/jtransform.rs RECOGNIZED_JITDRIVER_RECEIVER_ROOTS` ·
`optimizeopt/virtualize.rs optimize_setarrayitem_gc` (forces unless the array is
virtual) · `pyjitpl.py` vable access emits nothing ·
`majit-backend-dynasm/src/x86/assembler.rs` the two vtable modes ·
`majit-gc/src/header.rs` immortal restriction + align assert ·
`majit-metainterp/src/jitcode/mod.rs` `pub use majit_translate::insns` +
`pyjitpl/dispatch.rs` `JitCodeMachine` · `insns.rs` no `raw_load_r`/`raw_store_r`.
