# gh#346 — `dstrategy` trait-object field: lower `&dyn DictStrategy` to a base-classdef instance

## Problem

`w_dict_new` fails to lift with:

```
SomeInstance.setattr("dstrategy") generalize_attr failed:
UnionError: _ptr ∪ <other> — no upstream pair(s1, s2).union() handler in current subset
```

`W_DictObject.dstrategy` (dictmultiobject.rs:448) is typed `&'static dyn DictStrategy` — a
trait object. Its writes (dictmultiobject.rs:748/769/799/825/846) assign different concrete
strategy statics (`EMPTY_DICT_STRATEGY`, `OBJECT_DICT_STRATEGY`, …). The front-end lowers each
static read to a raw GCREF `_ptr` constant (via `ConstRefAddr` → `const_ref_gcref_constant`,
flowspace_adapter.rs:365) with NO class identity, and the `dstrategy` field itself projects to
`Impossible` (project_pyre_field_type has no `"dyn …"` arm, bookkeeper.rs:2600). The two
representations cannot union.

## Orthodoxy verdict (source-verified, 3 Explore agents)

The union machinery is **parity-correct**; there is no missing arm to add:

- `_ptr` = `SomeValue::Ptr(SomePtr)`, no classdef (lltype.rs:5954).
- `SomePtr ∪ SomeObject` = `UnionError` is ORTHODOX — RPython `llannotation.py:118-132` raises
  the same. Adding a `(Ptr, Instance)` arm would diverge; FORBIDDEN.
- `SomeInstance ∪ SomeInstance` already walks the base chain via `ClassDef::commonbase`
  (classdesc.rs:2431, a faithful port of RPython classdesc.py:251) and widens to the common
  ancestor.

In RPython, `W_DictObject.dstrategy = strategy` stores an ordinary instance; all strategies
subclass `class DictStrategy(object)` (pypy/objspace/std/dictmultiobject.py:462), so the
annotator unions the writes up to `DictStrategy` — no error. pyre's Rust `trait DictStrategy`
+ `impl DictStrategy for X` is the exact same shape.

**Orthodox fix = FRONT-END: lower the trait-object writes/reads to a base-classdef
`SomeInstance`, so the existing `commonbase` merge fires with zero union-layer change.**

## Existing machinery (already in tree, #346 S2.3)

`register_trait_family` (bookkeeper.rs:1956) already mints a base ClassDef for a trait and
interns each impl as a subclass with the base as its sole `__bases__` (same discipline as
enum-variant subclassing). It publishes the base in `pyre_trait_family_bases` keyed by the
trait's qualified `name_path()`, with accessors `registered_trait_family_base_root(leaf)`
(bookkeeper.rs:662, leaf→qualified, `None` if ambiguous) and `is_registered_trait_family_leaf`.

Registration is driven two ways:
- **config** `PipelineConfig.register_trait_families: Vec<String>` (pipeline.rs:58) — a list of
  trait qualified paths; harvested into `TraitFamilyRegistration` at lib.rs:1225, applied at
  codewriter.rs:158. **Empty in production** (test_support.rs:51). NOT gated.
- **auto** every `>=2`-impl trait, gated behind `PYRE_DYN_INDIRECT` (lib.rs:1264). The gate
  comment (lib.rs:1257) warns: minting base/impl subclass classdefs perturbs
  `pyre_struct_root_names` → `ensure_session` inheritance-id numbering even off-path, so the
  gate keeps prod byte-identical.

The machinery is wired only to the **parameter-seeding** path (`derive_subject_inputcells`
trait-family arm, flowspace_adapter.rs:2823) — a `&dyn Trait` function argument. It is NOT
wired to the **field** projection or the **static-constant** read.

## Plan — three hooks + one config line

### Hook 0 (config): register the DictStrategy family unconditionally

`test_support.rs:51` (the sole production `PipelineConfig` literal — despite its name it feeds
pyre-jit-trace via `#[path]` includes of call_spec/virtualizable_spec):

```rust
register_trait_families: vec!["pyre_object::dictmultiobject::DictStrategy".to_string()],
```

`trait_impl_owners` (lib.rs:1158, harvested from `concrete_trait_methods`) already maps this
qualified path to its 6 impl owners, so `make_registration` (lib.rs:1200) builds the family
with no further work. This runs through the NON-gated config path, so `PYRE_DYN_INDIRECT` is
untouched.

**RISK (measure first, in isolation):** the gate comment says config registration also mints
classdefs that shift `ensure_session` inheritance-id numbering. Whether one family shifts prod
numbering enough to regress is empirical. **Slice 0 = config line ALONE, then census set-diff +
check.py**, before adding hooks 1/2. If numbering shifts cause bit-exact regressions, this whole
direction needs the numbering made insertion-order-stable first (a larger sub-epic).

### Hook 1 (field side): `project_pyre_field_type` `"dyn <Trait>"` arm

bookkeeper.rs, before the `if !registered { return Impossible }` at :2600. When `stripped`
begins with `dyn ` (after also stripping a leading `'static`), take the trait leaf, resolve via
`registered_trait_family_base_root(leaf)`, and return `SomeInstance(base_classdef)`; else fall
through to `Impossible` (unchanged). Note `stripped` peels `&`/`mut`/`*const`/`*mut` but NOT
`'static` — the field string is `"dyn DictStrategy"` after `charon_dyn_trait_to_ast_string`
(mir.rs:12121 renders leaf-only, references stripped at mir.rs:11906), so verify the exact
stored spelling from the census/registry before matching.

### Hook 2 (constant side): `&'static dyn Trait` static read → `SomeInstance(impl subclass)`

`PlaceKind::Global` (mir.rs:4351). Today only the hard-wired `PyType` bucket narrows a static
to a classed instance (`pytype_static_addr` → `__pyre_cast_instance["PyType"]`, mir.rs:4379).
Generalize: when the static's declared type is a concrete `impl <RegisteredTrait>` struct,
wrap the `ConstRefAddr` in `__pyre_cast_instance[<impl_root>]` so its annotation lands as
`SomeInstance(impl_subclass_classdef)`, which unions cleanly (commonbase) with the
base-classed field cell from Hook 1.

## ⚠️ EMPIRICAL FINDINGS (7/15) — the 3-hook plan does NOT work as written

Slices built + census-verified. Results overturned the plan's assumptions:

- **Hook 0 (config) works, but is a no-op alone.** The production `PipelineConfig` is built in
  `pyre/pyre-jit-trace/build.rs:225` (NOT `test_support.rs` — that is test-only). Adding
  `"pyre_object::dictmultiobject::DictStrategy"` to `build.rs` `register_trait_families` DID
  register the family: DIAG dump shows `trait_family_registrations = [("pyre_object::
  dictmultiobject::DictStrategy", ["ModuleDictStrategy","BytesDictStrategy","EmptyDictStrategy",
  "EmptyKwargsDictStrategy","IntDictStrategy","ObjectDictStrategy","UnicodeDictStrategy",
  "IdentityDictStrategy","KwargsDictStrategy"]), …]` (9 impls harvested from
  `concrete_trait_methods`). check.py slice-0 stayed green (187/187/186, sole BASEFAIL). But the
  `dstrategy` census failure was UNCHANGED (`_ptr ∪ <other>`).

- **Hook 1 (`project_pyre_field_type` `"dyn <Trait>"` arm) NEVER FIRES.** Instrumented: no
  `[DIAG hook1]` and no `[DIAG ppft]` (function entry, filtered on "strategy"/"dyn") in a full
  census. `project_pyre_field_type` is simply never called with the `dstrategy` field type.

- **`project_struct_rows` is NEVER called for `W_DictObject`** either (`[DIAG psr]` = 0). So the
  pass-2 struct-row projection that would run `project_pyre_field_type` per field does not run
  for this struct at all.

**ACTUAL root cause (revised):** the `dstrategy` attr cell (`<other>`) is seeded by the
FORCE_ATTRIBUTES path, NOT the pass-2 struct-row projection:
`derive_program_metadata` (mir.rs:1093-1101) builds `struct_field_attrs` rows via
`tyref_to_attr_value_type(&f.ty)` (mir.rs:1098) — which maps `&dyn DictStrategy` to
`ValueType::Ref(None)` (its terminal fallback, no dyn-trait arm) → `register_struct_fields`
(lib.rs:619) → `FORCE_ATTRIBUTES_INTO_CLASSES` → `valuetype_to_someshell(Ref(None))`
(annotation_state.rs) = a **classdef-less `SomeInstance` shell** = the `<other>` operand.

**Architectural wall:** `valuetype_to_someshell(Ref(_))` DELIBERATELY ignores any root string
and returns a classdef-less instance (documented: a process-global bare-name→classdef lookup
would violate RPython's object-identity lltype cache). So even making `tyref_to_attr_value_type`
emit `Ref(Some("DictStrategy"))` would NOT attach the base classdef through this path. The real
classdef is meant to come from pass-2 (`project_struct_rows` → `project_pyre_field_type`), which
does not run for `W_DictObject`.

**Consequence:** the fix is NOT the 3 hooks below. It requires either (a) making `W_DictObject`
take the pass-2 struct-root projection path (so `project_pyre_field_type` runs per field, at
which point a `"dyn <Trait>"` arm resolving the registered family base would work — Hook 1 was
correctly designed, just never reached), or (b) a dyn-trait-aware seeding at the FORCE path that
can attach a classdef (fighting the documented object-identity constraint). Both are larger than
a localized front-end projection. All experimental edits (build.rs config, hook-1 field arm,
DIAG) were REVERTED; tree is clean. Family registration itself is sound and reusable once the
attr-cell path reaches a classdef-bearing projection.

## ✅ KEYSTONE RESOLVED (7/15) — ctor mints under the dotted qualname, a distinct cache key from the struct-root path

Traced + source-verified (2 agents + direct re-read of every cited site):

**The mint path.** `w_dict_new`'s body builds a `W_DictObject { ob_header, dstrategy, … }` struct
literal. The front-end lowers it to `CallTarget::SyntheticTransparentCtor { name, owner_path }`
(mir.rs:3881-3888). The flowspace adapter's `SyntheticTransparentCtor` arm, for a struct (not
enum-base, not closure), hits the terminal `else` at **flowspace_adapter.rs:2098-2099**:

```rust
let qualname = format!("{}.{}", owner_path.join("."), name);  // "pyre_object.dictmultiobject.W_DictObject"
bk.intern_class_by_qualname(&qualname)
```

`intern_class_by_qualname` mints the identity ClassDef but **never calls
`getuniqueclassdef_for_struct_root`** (bookkeeper.rs:2251-2348). Annotation of the class host then
routes `immutablevalue_hostobject` is_class (bookkeeper.rs:3429) → `ClassDesc::pycall`
(classdesc.rs:1392) → `getuniqueclassdef` → `_init_classdef` (classdesc.rs:1241), which fills attrs
**only** from `FORCE_ATTRIBUTES_INTO_CLASSES` (classdesc.rs:1280-1305). `project_struct_rows` — the
sole path with a per-field `project_pyre_field_type` a `"dyn <Trait>"` arm could hook — is reachable
ONLY from `getuniqueclassdef_for_struct_root` (pass-2 bookkeeper.rs:1858, drain :1874) and
`getuniqueclassdef_for_enum_variant` (:1921). None run here.

**Why the eager `ensure_session` struct-root prologue does not cover it** (the true keystone). The
prologue (pyre_call_registry.rs:562-566) DOES loop `pyre_struct_root_names()` → each is a `reg.fields`
key, which is the **`::`-spelled / bare-leaf** spelling (mir.rs:1041-1042). So it calls
`getuniqueclassdef_for_struct_root("pyre_object::dictmultiobject::W_DictObject")` — the **colon**
root. But `intern_class_by_qualname` keys its class cache on `canonical_struct_name(&cur)`
(bookkeeper.rs:2264). `canonical_struct_name` (descr.rs:440-449): a name containing `::` passes
through verbatim; a dotted name has no `::` and no `<`, misses `STRUCT_ORIGIN_REGISTRY`, and ALSO
passes through verbatim. So:

- ctor mint key = **`pyre_object.dictmultiobject.W_DictObject`** (dotted, verbatim)
- struct-root mint key = **`pyre_object::dictmultiobject::W_DictObject`** (colon, verbatim)

These are **two different `pyre_struct_root_classes` cache entries** = two distinct ClassDef
identities. The comment at bookkeeper.rs:2282-2284 documents this split as INTENTIONAL ("the classdef
cache key keeps the raw spelling, so this seeds the base without collapsing the ctor class onto the
`::`-spelled field-read class"). The census-visible classdef (the one the union error names) is the
**dotted, ctor-minted** one, whose `dstrategy` cell came from the FORCE shell = classdef-less
`SomeInstance(None)`. Pass-2 (if it runs at all) projects the **colon** classdef, never touching the
dotted one census shows.

**The actual union that fails** (re-verified model.rs:3206-3231). `SomeInstance ∪ SomeInstance` does
NOT error when one side is classdef-less — the `_ => None` arm (model.rs:3230) **widens to the
classdef-less top**. The `commonbase`-None UnionError fires only when BOTH sides carry a classdef.
So the real failing union `_ptr ∪ <other>` is `Ptr ∪ Instance(None-shell)` — the raw GCREF static
read (`_ptr`, no Instance form) cannot union with an Instance. This is an orthodox UnionError
(`SomePtr ∪ SomeObject`, llannotation.py:118-132), NOT a commonbase failure. **Corollary:** if the
static-constant reads were lowered to a *classed* `SomeInstance` (Hook 2), they would union cleanly
with the classdef-less FORCE shell via the `_ => None` widen arm — WITHOUT needing the field cell to
carry a classdef at all. This reframes the fix (see below).

### Revised fix options (source-grounded)

- **(A) Route the ctor mint through pass-2.** At flowspace_adapter.rs:2098-2099, before interning the
  dotted class, call `getuniqueclassdef_for_struct_root(&colon_root)` — mirroring the closure sub-arm
  one branch up (:2095). ⚠️ But the dotted-vs-colon cache-key split means the projected colon classdef
  is a DIFFERENT identity from the dotted ctor class; pass-2 would populate the wrong classdef unless
  the ctor ALSO interns under the colon key (collapsing the split the :2282-2284 comment guards). This
  is larger than "one line" and risks the documented ensure_session inheritance-id numbering shift.
- **(B) Hook 2 only (constant-side), NO field-side change.** Because the failing union is
  `Ptr ∪ Instance(None)` and `Instance(classed) ∪ Instance(None)` WIDENS (model.rs:3230), lowering the
  `&'static dyn Trait` static reads (mir.rs:4351 `PlaceKind::Global`) to a classed `SomeInstance` via
  `__pyre_cast_instance[<impl_root>]` may resolve the union with zero field-side / pass-2 work. The
  field cell stays classdef-less; the widen arm absorbs it. **This is now the most promising minimal
  slice** and sidesteps the entire dotted/colon keystone. Needs: confirm the static read currently
  produces `_ptr` (Ptr) not an Instance, and that `__pyre_cast_instance` on a static addr is wired.

**Next slice = probe (B):** instrument whether lowering just the static-constant reads flips
`_ptr ∪ <other>` to `Instance(impl) ∪ Instance(None)` → widen. If yes, (B) is the fix and the
dotted/colon split never needs touching.

## ✅ (B) LANDED (7/15) — resolves the `w_dict_new` setattr union; metric-neutral

Implemented in `front/mir.rs` `PlaceKind::Global` arm + helper `refs_static_zerofield_struct_root`.
A `refs`-bucket static whose declared type (`place.ty`, the bare pointee ADT for a Global place)
resolves to a **zero-field unit struct** is narrowed `ConstRefAddr → __pyre_cast_instance[<impl_root>]`
→ classed `SomeInstance(impl)`. Gated to zero-field structs so the 5 field-bearing object singletons
(`W_NoneObject`/`W_BoolObject`×2/`NotImplemented`/`Ellipsis`) in the same 13-entry `refs` bucket
(jit_fnaddr.rs:2062-2131) keep their raw-pointer lowering; catches the 8 dstrategy singletons.

**Result (verified):**
- The `SomeInstance.setattr("dstrategy")` `_ptr ∪ <other>` union in `w_dict_new` is **GONE**
  (census count present → 0). `store_global_value` advanced PAST the dstrategy union to a deeper
  pre-existing wall (`iter::adapters::enumerate::Enumerate::next` unregistered).
- **check.py bit-exact GREEN: dynasm 189/189, cranelift 189/189, wasm 188/188** (full pass).
  `cargo test -p majit-translate` green.
- **phaseA metric UNCHANGED at effect** (279→277 was a coincidental `typing_intrinsic_1/2` ±2
  census flicker from concurrent worktree builds, NOT the dstrategy paths): the graphs that hit the
  dstrategy union (`store_global_value`, `w_dict_new`) each have a CHAIN of further walls, so
  resolving one union does not yet drop the distinct-fail count. This is an orthodox, bit-exact
  correctness improvement (one union site resolved the RPython way) that lands as cleanup regardless
  of metric movement — the deeper walls (`Enumerate::next`, `IndexMap::new`, `Wtf8::as_str`, …) are
  separate registry-coverage slices.
- `setdict`/`setdictvalue` `_ptr ∪ <other>` is a SEPARATE pre-existing union site (phi-merge
  `mergeinputargs`, annrpython.rs:2161), present in baseline, untouched — `MAP_DICT_STRATEGY`
  (mapdict.rs:2766) is NOT in the `refs` bucket, so its read is not narrowed by this slice.

**Superseded open question (kept for history):** WHY does `getuniqueclassdef_for_struct_root("W_DictObject")`
/ `project_struct_rows` not run for `W_DictObject`, when its `ClassDef` IS minted? → ANSWERED above:
the ctor mints under a dotted cache key distinct from the colon struct-root key; pass-2 targets the
colon key, census shows the dotted one.

## Verification (bit-exact arbiter = check.py, 3 backends)

1. `cargo check -p majit-translate`.
2. **Slice 0 isolation:** config line only → clean-cache census build → set-diff vs current
   HEAD baseline (Option-2 tree). If numbering shifts add unrelated failures, STOP and reassess.
3. Full: hooks 1+2 → census set-diff (expect `w_dict_new` / `store_global_value` dstrategy
   failure gone, no new failures) + check.py dynasm/cranelift/wasm bit-exact (sole allowed
   fail = pre-existing `list_insert_pop_index` BASEFAIL).
4. Revert any slice whose bit-exact breaks.

## Scope note

`dstrategy` is the ONLY genuine trait-object polymorphic field among named pyre structs (the
other census "polymorphic-field" symptoms — `message`, `start`/`end`, `save_point`, `__pos_N` —
are four unrelated root causes: PyError classdef-identity dup, Rust stdlib `Range` monomorph
collapse, signedness, mir-walker synthetic positional fields). This epic is `dstrategy` alone;
the others are separate.
