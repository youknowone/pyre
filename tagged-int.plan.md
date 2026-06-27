# Tagged-int representation — design, disposition, and roadmap

Status: **design / not started.** This is the "Phase F tagged-int" referenced
by the four cast-boundary comments. The document below is the authoritative
plan; it supersedes the loose framing in those comments (notably the idea that
the casts themselves should perform the tag arithmetic — they must not; see §2).

The "tagged-int" representation stores a small Python `int` as an *immediate*
inside a `PyObjectRef` slot — an odd bit pattern `(value << 1) | 1` — instead of
a heap-allocated `W_IntObject`. Heap pointers are even-aligned, so the low bit
distinguishes an immediate from a real pointer. It is RPython's `UnboxedValue` /
`rerased` mechanism applied to `W_IntObject`.

---

## TL;DR / disposition

1. **The four cast boundaries stay identity reinterpret-casts forever.** They do
   **not** perform `<< 1 | 1` / `>> 1`. The `(i & 1) == 1` in their comments is
   an *assertion* that the operand is already an odd (tagged) value, not a
   tagging step. Tagging arithmetic lives **only** at the box/erase site. The
   earlier "flip the four casts to tagged semantics" framing is a deviation
   from upstream and must not be implemented (§2, evidence).

2. **Tagged-int is an OPTIONAL RPython feature, off by default in PyPy.**
   `translationoption.py:185 BoolOption("taggedpointers")` defaults false;
   `W_IntObject` is a plain heap box (`intobject.py:540-542 __slots__='intval'`);
   there are **zero** production `UnboxedValue` subclasses. pyre's current
   heap-boxed `W_IntObject` + `withprebuiltint` cache (`intobject.rs`) already
   mirrors the PyPy **default** line-for-line. So "tagged-int not implemented"
   is faithful to the default, **not** a parity gap. Pursue it only as a
   deliberate, optional optimization — not as a correctness obligation.

3. **The runtime *enablement* (making `w_int_new` return a tagged immediate) is
   genuinely gated on #73** (symbolic-valuestack / pc-map). A tagged immediate
   reaching a deopt slot typed `Type::Ref` is dereferenced as a pointer by the
   blackhole and crashes (§4, evidence). This half cannot land green today.

4. **The structural, ungated half is landable** behind a compile-const flag
   defaulting OFF: the primitive (`tag/untag/is_tagged`), the tag-aware reader
   and dispatch chokepoints, and (carefully) the cast assertions. All inert
   until enablement. See the slice roadmap (§6).

5. **The genuinely faithful, ungated, behavior-improving work hiding adjacent to
   this epic is NOT tagging at all** — it is porting `int.__eq__`-by-value for
   `is_w` and the `IDTAG`-based `immutable_unique_id` (§7). pyre is currently
   missing both. These are independent of tagging and of #73. They carry a real
   behavioral change (`big is big` → True, PyPy semantics) and so need check.py
   adjudication, but they are the actual int-identity parity gap.

---

## 1. Representation (the bit layout, if pursued)

Authoritative spec is the already-ported rtyper helper
`majit/majit-translate/src/translator/rtyper/lltypesystem/rtagged.rs`
(`ll_int_to_unboxed` = `value * 2 + 1` with `checked_mul`/`checked_add`;
`ll_unboxed_to_int` = `n >> 1`; `is_unboxed_instance` = `(n & 1) != 0`),
itself a port of `rpython/rtyper/lltypesystem/rtagged.py` /
`rpython/rlib/rerased.py`.

A new runtime module `pyre/pyre-object/src/tagged_int.rs` would mirror it:

```rust
fn tag_int(v: i64) -> PyObjectRef     // ((v << 1) | 1) as *mut PyObject; caller range-checks
fn untag_int(p: PyObjectRef) -> i64   // (p as i64) >> 1   (arithmetic, sign-preserving)
fn is_tagged_int(p: PyObjectRef) -> bool  // (p as i64) & 1 == 1
fn fits_tagged(v: i64) -> bool        // v >= (i64::MIN >> 1) && v <= (i64::MAX >> 1)
```

Soundness of the tag bit: every real `PyObjectRef` comes from
`Box::into_raw(Box::new(W_IntObject))` (`lltype.rs`) or
`gc_hook::try_gc_alloc_stable`, both 8-byte aligned (`W_IntObject` carries an
`i64`), so the low 3 bits are always zero on a genuine heap pointer. The odd
tag therefore never collides with a real pointer. `None`/`True`/`False` are
8-byte-aligned statics (even) and are **not** tagged (PyPy tags `int` only).

---

## 2. The casts stay identity (do NOT flip them) — decisive

`rpython/jit/metainterp/blackhole.py:603-610`:

```python
def bhimpl_cast_ptr_to_int(a):
    i = lltype.cast_ptr_to_int(a)
    ll_assert((i & 1) == 1, "bhimpl_cast_ptr_to_int: not an odd int")
    return i                       # NO >> 1
def bhimpl_cast_int_to_ptr(i):
    ll_assert((i & 1) == 1, "bhimpl_cast_int_to_ptr: not an odd int")
    return lltype.cast_int_to_ptr(llmemory.GCREF, i)   # NO << 1 | 1
```

The cast is a pure bit reinterpretation; the value is *already* `v*2+1`
(tagged earlier, at erase/box time). The backend agrees: x86
`genop_cast_*` = `_genop_same_as` (identity); `runner_test.py:1957` round-trips
`cast_int_to_ptr(-17) -> cast_ptr_to_int == -17` (an even value) through the
casts *identically*. pyre's cranelift backend already documents and enforces
this (`majit/majit-backend-cranelift/src/compiler.rs:9210-9222`: casts wired to
`_genop_same_as`; "OR-tagging here would fold a fake odd pointer ... could
collide with a real GC pointer").

Consequence for the four boundaries (all currently identity):
- `majit/majit-metainterp/src/executor.rs:758-759` — const-fold: **keep identity**.
- `majit/majit-metainterp/src/blackhole.rs:8131-8144` — bhimpl: **keep identity**;
  the only change ever is adding the `(i & 1) == 1` *assertion*.
- `pyre/pyre-jit-trace/src/jitcode_dispatch.rs:17082/17096` — recorder: **keep identity**.

Why the JIT needs no runtime tag check at the casts: each `OpCode` has a static
result `Type` (`CastPtrToInt ∈ int!`, `CastIntToPtr ∈ ref_!`), and the blackhole
keeps three physically separate register banks (`registers_i/r/f`); the cast
handlers move bits by argcode alone. Int-vs-ref is compile-time at the JIT layer.

**Boundary #5 (do not forget):** the optimizer registers `CastPtrToInt` /
`CastIntToPtr` as mutual pure inverses for CSE
(`majit/majit-metainterp/src/optimizeopt/rewrite.rs:1926-1941`). This is sound
*only because the casts are identity*. It is another reason never to put
arithmetic in the casts.

---

## 3. Runtime consumer chokepoints (tag-aware sites, if enabled)

The runtime funnels through very few sites:

- **Box (maker):** `pyre/pyre-object/src/intobject.rs:90 w_int_new` — the single
  tag site (`if fits_tagged(v) return tag_int(v)` before the malloc/gc block).
  `w_int_new_unique` (subclass identity) must **never** tag.
- **Unbox (reader):** `intobject.rs:141 w_int_get_value` (branch on
  `is_tagged_int`); the one direct-deref reader
  `pyre/pyre-interpreter/src/display.rs:242` must route through it.
  (`display.rs:257` is the **bool** branch — leave it.)
- **Type dispatch:** `pyre/pyre-object/src/pyobject.rs` `is_int` must
  short-circuit true on `is_tagged` *before* the `ob_type` deref;
  `ll_type`/`ll_inst_type` synthesize `&INT_TYPE` for a tagged value. Mirror
  `rtagged` `ll_unboxed_getclass`. Note `is_int` today already includes
  `BOOL_TYPE` — reconcile. Per upstream (`rtagged.py:64-96 gettype_from_unboxed`)
  the `& 1` check is gated on a static `can_be_tagged`; do not sprinkle an
  unconditional runtime `& 1` into the shared dispatch for every object
  (`rerased.py:1-3`: the point is to avoid "tag checks everywhere").
- **repr/str:** `display.rs py_repr/py_str` must format the immediate before the
  `ob_type` deref.
- **Arithmetic fast paths** (`descroperation.rs` int_add/...): no per-op change;
  they ride free once read+make are tag-aware. Verify-only.

---

## 4. The #73 gate on enablement — decisive

At deopt, every reconstructed value-stack slot is routed to the int-bank or
ref-bank by `slot_types: Vec<Type>` (`majit/majit-metainterp/src/resume.rs`),
which is keyed off the Python pc / pc-map kept-stack — the #73
symbolic-valuestack layer. `resume.rs` already calls `box_int` for a `Type::Ref`
slot that the optimizer unboxed to `Int`. The instant `w_int_new` can return a
tagged immediate that flows into a `Type::Ref` slot, deopt pushes a tagged value
into `registers_r` and the blackhole dereferences it as a pointer → crash.

So the enablement flip (`w_int_new` tagging live; Slice 6) cannot land green
until #73 lets `slot_types` / symbolic-valuestack carry "this Ref-slot may hold a
tagged immediate." This confirms the 2026-06-24 verdict as **PARTIAL**: the
structural half is ungated; the enablement half is #73-gated.

Orthogonal to #424: that bug is the `W_IntObject*`-in-a-kept-slot
Python-object layer (heap box); tagging is the `Signed<->Ptr` lltype layer.
Tagging does not fix #424's depth>1 reconstruction.

---

## 5. The GC consumer-side requirement (missed by the naive design)

Guarding the *producer* (`w_int_new`) is not enough. Once a tagged immediate is
stored into any traced container (list/tuple item, dict value, instance dict,
closure cell), the collector reads it back out of a `GcRef` field and would
dereference it as a header. Every collector consumer site in
`majit/majit-gc/src/collector.rs` (the fixed-field and varsize trace loops,
`mark_object`, `copy_nursery_object`, weakref target reads) filters only on
`is_null` + an address-range test (`is_in_nursery` / `is_managed_heap_object`);
**none** checks the low bit. Because `fits_tagged` admits the full 63-bit value
range, a tagged immediate's bit pattern can land inside the live heap range,
pass the range check, and have `header_of(addr) = addr - 8` read garbage — a
value/ASLR-dependent SIGSEGV.

Therefore enablement (Slice 6) must be preceded by a collector slice that adds an
`is_tagged`-skip (`field & 1 == 1`) at every collector consumer site (mirror
gctransform: odd pointers are not traced). This is a hard prerequisite, not an
optimization.

---

## 6. Slice roadmap

Each slice is independently gate-green (census neutral-or-shrink AND check.py
both backends) and inert (behind `PYRE_TAGGED_INT=false`) until Slice 6.

| # | Title | blocked_by | Notes |
|---|-------|-----------|-------|
| 0 | Design doc (this file) + inert `tagged_int.rs` primitive + unit tests | none | zero call sites; census neutral by construction |
| 1 | Tag-aware `w_int_get_value` + route `display.rs:242` through it | #0 | dead branch (no tagged value exists yet) |
| 2 | `is_int` short-circuit + `ll_type`/`ll_inst_type` synth (gated on static can_be_tagged) | #1 | dead branch |
| 3 | repr/str tag-guard | #2 | dead branch |
| 4 | GC collector consumer `is_tagged`-skip (collector.rs) | #0 | hard prerequisite for #6; inert until a tagged value exists |
| 5 | Add `(i & 1) == 1` *assertions* to the cast bhimpls (casts stay identity) + CSE inverse audit | #0 | assertions only; backend untouched |
| 6 | **ENABLE** tagging in `w_int_new` | **#73**, #1-5 | live flip; reds on every polymorphic-int-slot deopt until #73 |

The "first slice" for *starting* PART 1 is **Slice 0 (this doc)**. The inert
primitive itself is deliberately **deferred**, not landed with the doc, because
(a) the epic is an optional feature whose enablement is #73-gated, and (b) the
GC-consumer requirement (§5) and the assertion-only cast change (§5/§2) should be
designed before committing infra. Land code only when the user elects to pursue
the optional epic.

---

## 7. The real ungated int-identity parity gap (separate from tagging)

PyPy implements int `is` and `id()` at the objspace layer, independent of any
pointer tagging:

- `intobject.py:553-567 W_IntObject.is_w` compares **by value** (with a
  `user_overridden_class` guard) — so in PyPy `big is big` is **True**.
- `intobject.py:55-60 immutable_unique_id` = `bigint << IDTAG_SHIFT | IDTAG_INT`
  (`util.py` `IDTAG_SHIFT=4`, `IDTAG_INT=1`) — a value-derived id, so equal ints
  share `id()`.

pyre is missing both: `baseobjspace.rs:2239 is_w` is bare `std::ptr::eq`, and
`function.rs:1642 immutable_unique_id` is `_obj as usize` (raw pointer). So
pyre's `big is big` is False and `id(big)` is the malloc address — a real
deviation from PyPy.

Porting these is faithful, ungated, and touches neither the cast layer nor #73.
The caveat is behavioral: `big is big → True` and value-shaped `id()` may flip
check.py tests that pin CPython-style int identity. This makes it a deliberate
slice needing check.py adjudication, but it is the actual int-identity parity
work — and a more valuable, more faithful target than the optional tagged-int
enablement, which is blocked on #73 regardless.

---

## 8. False-start traps (NOT tagged-int infra — exclude)

- **Register-class** `blackhole.rs:7429-7461` `/id>X` / `/iXd` handler variants
  ("pyre tagged-int base in an int register"): a regalloc bank selector for an
  already-real even-aligned base pointer. Not live value tagging.
- **Resume byte-stream** `resume.rs` `tag(value, tagbits) = (value << 2) | tagbits`
  with `TAGCONST/TAGINT/TAGBOX/TAGVIRTUAL`: a 2-bit serialization tag in the
  resume metadata, decoded to a typed `Const::Int`. Different layout (`<< 2`),
  different purpose. Do not reuse `TAGINT` or the `<< 2` encoder for the runtime
  representation.
