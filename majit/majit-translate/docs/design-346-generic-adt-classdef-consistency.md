# Design: #346 generic-ADT ClassDef consistency (C-cluster)

**Status:** scoping / design. Not yet implemented.
**Branch:** `rtyper-legacy` (fixed per project rule).
**Arbiter:** census set-diff (`PREPASS histogram phaseA`) + `python3 ./pyre/check.py` (bit-exact, all 3 backends).

## 1. What this epic closes

gh#346 retires the rtyper legacy walker (`cutover::is_known_unported`, cutover.rs:1165, must match
nothing). Census on merged main (`38173e3cde6`) = **277 PREPASS-fail paths**. The census's own orthodox
histogram buckets the single largest *rtyper-stage* root cause as the generic-ADT collapse:

| bucket | count | note |
|---|---|---|
| UNION-PAIR-PORT | 44 | `setattr(field)` heterogeneous union, single collapsed classdef field |
| GENERIC-ADT-SPECIALIZE | 9 | `cannot unify instances with no common base` / `don't know how to convert` |
| UNION-ERROR (generic-ADT phi / pair) | 4 | mergeinputargs + `setbinding` monotonicity PANIC |

= **57 directly bucketed**, but the actual reach is wider: **58 distinct roots** touch a
UnionError/setbinding/no-common-base failure (store_global/store_name, the whole unary/binop
descroperation family, call_function, setattr_str, w_dict_new, Range build, PyError::new). A large share
of the 28 `BLOCKED-BLOCK` (downstream dead-end) paths are the hydra tails of these same union failures,
so the domino payoff plausibly exceeds 57.

## 2. Root cause (source-verified, three-agent cross-check)

### 2a. The failure is INCONSISTENT splitting, not absent splitting

pyre already implements *partial* per-instantiation ClassDef splitting (retrofit under #100/#312/#346).
A generic ADT's ClassDef name is the bare Charon `name_path()` (`ullbc.rs:293` — generic args live in
use-site `generics.types`, never in the decl name), **optionally** suffixed with `<T0,T1,…>` by ONE
predicate:

- `adt_head_instantiation_suffix` (mir.rs:12384) returns `Some("<…>")` **iff** the head is an
  `TypeDeclKind::Enum` (mir.rs:12390 — **structs always collapse**) AND every rendered type-arg passes
- `type_arg_splits_per_instantiation` (mir.rs:12361): `false` for `DEFERRED = ["f32","f64","()",""]`.

ClassDef identity is the canonical name STRING (`classdesc.rs:2198` — `struct ClassDef` carries
`name: String`, **no generic-args field**; dedup by `canonical_struct_name`, descr.rs:429). So bare
`Option` and suffixed `Option<Result<*mut PyObject,PyError>>` are **distinct ClassDefs**.

### 2b. The four projection sites disagree

Four independent sites project a TyRef→class-root string; they do NOT all route through the suffix gate:

| site | fn | mir.rs | suffix? |
|---|---|---|---|
| A receiver/input | `adt_node_class_root` | :11380 keeps / :11383 bare | enum+gate only; **struct always bare** (:11367 core/std/alloc → `None`) |
| B constructor (`Some`/`Ok`) | `resolve_aggregate_adt` | :4569 keeps / :4570 bare | enum+gate |
| C variant field read (`.0`) | `resolve_adt_field` | :4636 keeps / :4637 bare | enum+gate |
| **D discriminant read** | `tyref_adt_name_path` | :3941 / :8989 | **NEVER suffixed — always bare** |

Site D (`Rvalue::Discriminant`, mir.rs:3938) reads the enum tag through raw `name_path()` with no suffix
branch at all. So a match on `Option<Result<..>>` reads the tag as bare `Option::Some` (D) but the
constructor wrote suffixed `Option<Result<..>>::Some` (B). These are distinct ClassDefs.

### 2c. The three surfaced symptoms all trace to 2b (agent-3 raise-site map)

1. **`setbinding: new value does not contain old` PANIC** (annrpython.rs:585): the fixpoint monotonicity
   assert. One pass binds the suffixed variant, another the bare variant; `contains` = `union==self`
   (model.rs:3513); `commonbase(bare, suffixed)` (classdesc.rs:2431) returns none or a shared ancestor
   ≠ self → `contains` false → PANIC. Hitters: `unary_invert_value`, `unary_negative_value`.
2. **`cannot unify instances with no common base class`** (model.rs:3216, mirrors binaryop.py:672) and
   **`don't know how to convert from … to …`** (rtyper.rs:5775 via rclass.rs:4377 `Ok(None)` — sibling
   InstanceReprs are neither's `basedef`). Both bucketed GENERIC-ADT-SPECIALIZE.
3. **`setattr(field) generalize_attr failed: UnionError: _ptr ∪ <other>`** (unaryop.rs:4302 →
   model.rs:3424 `_ =>` fallback). This is the **struct** collapse (site A always bare for structs):
   `W_DictObject.dstrategy` gets `_ptr` at one site and another tag elsewhere onto ONE `Attribute`
   (classdesc.rs:577 `Attribute::merge` / :564 `add_constant_source`, both call `model::union`). Same for
   `PyError.message` (`str ∪ …`), `Range.start/end` (`int ∪ …`).

### 2d. ⚠️ ORTHODOXY VERDICT — the naive reading of the bucket name is NON-orthodox

The census bucket is literally named `per-instantiation classdef`, which invites "give every
instantiation its own classdef." **RPython forbids this** (agent-2, vendored source):

- Class specialization was **explicitly removed**: `rpython/annotator/classdesc.py:507-510` raises
  `"Class specialization has been removed"` for a `_annspecialcase_` class tag. `getclassdef(key)`
  **ignores the key** and returns the unique classdef (classdesc.py:669). RPython = **exactly one
  ClassDef per class, with a single generalized (union'd) field** (classdesc.py:87-101 `unionof`).
- The union-requires-common-base rule (binaryop.py:664-683) and the heterogeneous-union raise are
  therefore **orthodox** — RPython raises on `str ∪ int` too. The missing `model.rs:3424` arm is NOT a
  bug to fill.
- RPython keeps `Option<A>`/`Option<B>` distinct via **function specialization of the factory**
  (`specialize:argtype` key=arg `knowntype`, specialize.py:356; or `specialize:call_location` key=call
  op, specialize.py:368) — producing distinct *function graphs* that reference distinct *concrete
  classes*, OR via **container-style per-position defs** (lists/dicts get one `ListDef` per creation site
  keyed by `position_key`, bookkeeper.py:178; repr keyed on `listitem`, rlist.py:59). User classes get
  ONE classdef; built-in containers get per-site defs.

**Implication:** `Option`/`Result` are RPython *built-in-container-like* (per-site distinct payload
repr), NOT user-class-like (one classdef). pyre's existing `<…>` suffix IS the container-style per-site
model — it is orthodox *in spirit* (each instantiation = its own payload-typed pseudo-container). The
epic is therefore **not** "add specialization"; it is **"make the existing per-instantiation split
total and consistent so bare and suffixed spellings never collide, and give siblings a shared base so
they still union to the widened parent."**

## 3. Two candidate directions

### Direction 1 — TOTALIZE the suffix (make all four sites agree, always split)

Route sites A(struct)/D(discriminant) through `adt_head_instantiation_suffix` too, and remove the
`DEFERRED`/struct/core-std-alloc collapse gates so *every* generic instantiation splits. Then wire every
suffixed variant under its bare enum base (bookkeeper.rs:2028-2031 already does this for the constructor
path) so `commonbase` succeeds and unions widen to the parent.

- **Pro:** conceptually uniform; kills 2c.1 and 2c.2 at the source.
- **Con / RISK:** structs (2c.3, the 44-count UNION-PAIR-PORT bucket) are the *biggest* sub-cluster and
  the suffix machinery is enum-only today (mir.rs:12390). Extending it to structs is new ground —
  `W_DictObject` is not generic, so its `dstrategy` heterogeneity is NOT an instantiation split at all;
  it is a genuinely polymorphic field (`_ptr` = a dict-strategy object pointer). Splitting won't help;
  that field needs a real union arm OR a typed-Ref field projection. **So Direction 1 does NOT address
  the 44-count struct bucket** — those are a different problem wearing the same error string.

### Direction 2 — RECONCILE bare↔suffixed (make the collision impossible, keep gates)

Keep the enum-only suffix but guarantee bare and suffixed spellings of the same variant are always
mutually unifiable: make the **bare** spelling the shared base of **every** suffixed spelling (it already
is for the constructor path via `intern_enum_variant_host`, bookkeeper.rs:2028), and fix site D
(discriminant) to read the SAME spelling the constructor wrote — either by suffixing D, or by having the
union/commonbase treat bare-stem as the universal base of all its suffixed instantiations.

- **Pro:** narrow; directly kills the `setbinding` PANIC (2c.1) and the sibling-convert failures (2c.2)
  which are the `Option`/`Result` enum cluster (GENERIC-ADT-SPECIALIZE 9 + UNION-ERROR 4 = 13).
- **Con:** does not touch the 44-count struct UNION-PAIR-PORT bucket either (same reason as Direction 1).

### The struct bucket (44) is a SEPARATE problem

Both directions reveal that the 44-count `UNION-PAIR-PORT` struct-field bucket is NOT the generic-ADT
collapse — it is **polymorphic struct fields** (`W_DictObject.dstrategy: *mut <strategy>`,
`PyError.message`, `Range.start/end`) where one field legitimately holds different types. RPython's
answer here is either (a) the field is genuinely one type (pyre mis-lowers a Rust enum/trait-object field
to a raw `_ptr` where RPython would carry a `SomeInstance`), or (b) a real common-base union. This needs
its own investigation before it can be scoped — it is likely the typed-Ref ClassDef projection epic
(the "B" work) wearing a union-error mask, NOT this one.

## 4. Recommended scope split

1. **This epic (C-enum, ~13 paths + downstream domino):** Direction 2. Reconcile bare↔suffixed for
   `Option`/`Result`. Concretely: (i) fix site D discriminant to carry the instantiation suffix (or
   canonicalize it to the bare base that is the union parent); (ii) verify every suffixed variant chains
   `basedef` to the bare enum base so `commonbase` widens instead of failing; (iii) confirm the
   `setbinding` PANIC and sibling-convert failures clear. Empirical arbiter: census set-diff + check.py.
2. **Separate follow-up (struct polymorphic-field, ~44 paths):** investigate `dstrategy`/`message`/
   `start`/`end` field lowering. Likely the typed-Ref field-projection epic, not generic-ADT. Do NOT
   fold into this epic — the shared error string is misleading.

## 5. Load-bearing file:line index

- Suffix gate: mir.rs:12361 (`type_arg_splits_per_instantiation`), :12384 (`adt_head_instantiation_suffix`).
- 4 projection sites: A mir.rs:11380/11383, B mir.rs:4569/4570, C mir.rs:4636/4637, **D mir.rs:3938-3946
  + :8989 (never suffixed)**.
- ClassDef identity: classdesc.rs:2198 (`struct ClassDef`, name-only), descr.rs:429 (`canonical_struct_name`).
- Variant base wiring: bookkeeper.rs:2023-2031 (`intern_enum_variant_host`, base chain), :1897
  (`getuniqueclassdef_for_enum_variant`), mir.rs:607-612 (per-instantiation variant row pre-registration).
- Union / commonbase: model.rs:3196-3239 (Instance∪Instance), :3216 (no-common-base raise), :3424
  (`_ =>` heterogeneous fallback), classdesc.rs:2431 (`commonbase`).
- setattr generalize: unaryop.rs:4217-4319 (:4302 panic), classdesc.rs:564/577 (`Attribute` union).
- Convert / fixpoint: rtyper.rs:5775 (convert raise), rclass.rs:4338-4378 (InstanceRepr convert),
  annrpython.rs:581-585 (`setbinding` monotonicity PANIC), model.rs:3513 (`contains`).
- RPython orthodoxy: classdesc.py:507 (class-spec removed), :669 (getclassdef ignores key), binaryop.py:664
  (Instance union common-base), specialize.py:356/368 (argtype/call_location), bookkeeper.py:178 (per-site
  ListDef), rlist.py:59 (repr keyed on listitem).

## 6. Honesty notes

- The bucket name `per-instantiation classdef` is misleading; the orthodox model is container-style
  per-site payload typing (which pyre already half-does), NOT class specialization (RPython-forbidden).
- The 57 "C" paths are really ~13 enum (this epic) + ~44 struct polymorphic-field (different epic). Only
  the ~13 are cleanly in scope; claiming 57 would overstate.
- Census set-diff MISSES body-level regressions in already-lifting graphs; check.py is the only guard
  for those. Any implementation MUST run full check.py (bit-exact, 3 backends), not census alone.
