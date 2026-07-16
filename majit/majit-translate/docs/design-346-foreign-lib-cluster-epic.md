# gh#346 — Foreign-std-library cluster epic (vec! / malachite / String / Wtf8 / IndexMap)

## Why this epic exists

On base `eca75827fe4` the JIT-prepass census has **268 distinct `[PREPASS phaseA fail]`** paths.
A root-cause analysis (deepest-root, per-record all-blockers) ranked the leaves by "sole-unblock"
leverage. The top three (`box_assume_init` / vec!, `malachite ::try_from`, and the
`Wtf8`/`Map::collect`/`IndexMap` group) looked independently high-value, but a **peel-and-recensus**
of each proved otherwise:

- vec!/NewList recognizer re-add → census only 268→266 (net −2), and it **breaks the build**
  (`newlist/r>r` unwired-snapshot).
- `malachite try_from` via `plain_int_w` `#[dont_look_inside]` → **net ZERO** (and the attribute
  didn't even take: `plain_int_w` is inlined by rustc/Charon before the JIT sees it).
- Combined vec!+try_from → 268→265, only `getitem_list` lifts.

**Root insight:** every hot list/dict/str *mutation* graph (`setitem_list`, `getitem_list`,
`w_list_setitem`, `setitem_bytearray`, `plain_int_w`) is guarded by a **stack** of foreign-std walls.
Peeling one exposes the next:

```
box_assume_init (vec!) → malachite ::try_from → String::new → core::slice::get / join
  → Wtf8Buf::with_capacity → Enumerate::next → IndexMap::insert / get
```

No *single* leaf yields standalone census lift because the leaves **co-block the same graphs**. The
only way to lift the cluster's hot graphs is to close the whole wall-stack **together**. This is the
"census-line depth is a LIAR" phenomenon (documented in the rtyper-legacy rebase memory) at cluster
scale — the census reports only the first wall per graph.

## Slice ORDER — corrected 7/16 (the recognizer is a CAPSTONE, not a prerequisite)

The original plan put vec!/NewList first because `box_assume_init` is the *census-visible frontier*
wall. That was wrong, and it was proven wrong empirically (see "Verdict" below). The front/mir vec!
recognizer is a **shared front-end rewrite**: it plants `OpKind::NewList` into a graph regardless of
whether that graph will lift through the two-phase prepass or drop to the legacy walker. The legacy
walker cannot lower `NewList` (it never runs `rtype_newlist`), so its raw op reaches the assembler's
default arm and emits `newlist/r>r` — an opname with no blackhole handler — which breaks the build via
`default_bh_builder_unwired_set_matches_task_85_snapshot`. Therefore the recognizer is only safe once
**no** vec!-bearing graph can still drop to legacy, i.e. after every co-blocking wall closes.

**Corrected order: Slice A (try_from) → Slice B (String/Wtf8/IndexMap/slice/iter) → Slice C
(vec!/NewList recognizer, LAST).**

### Slice A — malachite `try_from` (task #21, FIRST, no deps)

**Census (base `ccdc1a52be2`):** 16 phaseA graphs have a `try_from` first wall, split into TWO
unrelated families:
- **13 graphs = malachite** (`["malachite_bigint","bigint","<Impl>","try_from"]`) — the Slice-A target.
- **3 graphs = int-to-int** (`["core","convert","num","<Impl>","try_from"]`: `int_pow`, `pow`,
  `opcode_get_iter`) — NOT malachite. `int_pow` is `u32::try_from(vb)` with a genuine
  `Err(_)=>return Err(memory_error)` branch (descroperation.rs:860). Out of Slice A's scope; defer.

**Reachability of the 13 malachite graphs** (traced through the census onion): every one reaches the
wall through exactly one of three functions — `setitem_list` (baseobjspace.rs:2359), `getitem_list`
(baseobjspace.rs:1268), or `plain_int_w` (listobject.rs:362, reached via `w_list_setitem` /
`w_list_append` / directly).

**The crux — the narrowing is NOT uniform (verified by direct `rg` census of every
`i64::try_from(w_long_get_value(...))` site):**
- **bucket (a) panic** — `plain_int_w` (listobject.rs:366) is `.unwrap_or_else(|_| panic!())`,
  semantically identical to `jit_bigint_to_i64_value` (longobject.rs:364). A pure target-swap is
  bit-exact HERE.
- **bucket (c) genuinely-fallible** — `getitem_list` (baseobjspace.rs:1303) and `setitem_list`
  (baseobjspace.rs:2372) — and 6 more `getindex_w`-inlined siblings (baseobjspace.rs:1362, 1427,
  1493, 2523, 2547, 2584; 8 total) — are `match i64::try_from(...) { Ok(i)=>i, Err(_)=>return
  Err(IndexError/ValueError) }`. Overflow throws a **Python exception**, NOT a panic. A panic-residual
  swap would be a CORRECTNESS REGRESSION (bit-exact violation). The coercion is deliberately inlined
  per-callsite ("the same rtyper reason as getitem_list") — no shared choke point.

So Slice A is NOT the "pure target-swap" the original note assumed. Two distinct lowerings:

1. **bucket (a) `plain_int_w`**: pure target-swap of `i64::try_from(<opaque BigInt>)` →
   `jit_bigint_to_i64_value` (both `#[dont_look_inside]` residuals ALREADY registered,
   jit_fnaddr.rs:911-919), guarded on `tyref_is_opaque_bigint` (mir.rs:11226). Mirror
   `bigint_binop_residual_path` (mir.rs:6457). But the enclosing `Result` local — `plain_int_w`'s
   `try_from` result is immediately `.unwrap_or_else`d, so the MIR may already fold it; verify.
2. **bucket (c) `match i64::try_from(<opaque BigInt>)`** (8 sites, ALL verified by an exhaustive
   census: `getitem_list` baseobjspace.rs:1303, `getitem_str` :1362, `getitem_bytes_like` :1427 +
   :1493, `setitem_list` :2372, `setitem_bytearray` :2523 + :2547, `byte_w` :2584 — each
   `Ok=>i, Err(_)=>return Err(IndexError/ValueError)`): lower to the runtime-discriminant `Result`
   aggregate, using `try_lower_checked_neg` (mir.rs:8610) as the EXACT template (it already builds a
   runtime-disc `Result`/`Option` via `emit_tagged_pair_aggregate`, mir.rs:9339). Emit:
   `fits = jit_bigint_to_i64_fits(bigint)` (residual, returns 1 when it fits), `disc =
   BinOp("eq", fits, 0)` (Result convention: **Ok=0, Err=1** — mir.rs:8259, so disc=0 when it fits),
   a `payload`, then `emit_tagged_pair_aggregate(disc, payload)` with `Ok`=tag 0. The
   `Err(_)=>return Err(IndexError)` arm survives UNTOUCHED as real user code reached via the disc==1
   switch branch. This is the substantive part of Slice A.

   **⚠️ EAGER-PAYLOAD PANIC HAZARD (the one non-obvious constraint):** `emit_tagged_pair_aggregate`
   writes `__pos_0 = payload` UNCONDITIONALLY in the call's block (mir.rs:8682-8689 pushes the payload
   op before the goto), *before* the consumer's discriminant switch runs in a successor block. For
   `checked_neg` the payload op is `neg` (total, never traps), so eager eval is safe. But
   `jit_bigint_to_i64_value` **PANICS on overflow** (longobject.rs:366). If `payload =
   jit_bigint_to_i64_value(bigint)` is computed eagerly and the BigInt does NOT fit, the residual
   panics in the walker/blackhole graph BEFORE the switch can route to the `Err` arm — turning
   `lst[2**100]` into a panic instead of the correct `IndexError`. **A bit-exact regression.**
   Resolution: `__pos_0` is only READ on the Ok path (disc==0 guarantees fits), so its value on the
   Err path is dead — the payload op only needs to be TOTAL (non-trapping), not correct-on-overflow.
   Use a NON-PANICKING total residual for the payload — add `jit_bigint_to_i64_value_or_zero`
   (`i64::try_from(num).unwrap_or(0)`, mirrors the existing bucket-(d) idiom at listobject.rs:1317)
   next to the fits/value pair in longobject.rs and register it in jit_fnaddr.rs. On the Ok path it
   equals `jit_bigint_to_i64_value`; on the (dead) Err path it returns 0 instead of panicking. This is
   RPython-faithful: upstream `toint`-after-`fits_int`-guard is elidable *because* the guard proves it
   fits; with no guard in the interpreter graph, the total form is the honest lowering.

   For **bucket (a) `plain_int_w`**: the same aggregate lowering serves it — its `.unwrap_or_else(|_|
   panic!)` reads `__pos_0` on the (always-taken, precondition-guaranteed) Ok path. Prefer UNIFYING
   both buckets on one recognizer for `<opaque BigInt>::try_from` rather than a separate target-swap.

Fail-safe by construction (a non-BigInt / unlisted target leaves the residual `<Impl>` call the census
Skips). Verify: `cargo test -p majit-translate` + census set-diff (expect the 13 malachite graphs to
drop OR expose their NEXT co-blocking wall) + `default_bh_builder_unwired_set_matches_task_85_snapshot`
green + `check.py` bit-exact 3-backend (list/dict subscript with huge int indices exercises the
Err arm — MUST still raise IndexError/ValueError, not panic).

### Slice B — String / Wtf8 / IndexMap / slice / iter-adapter residuals (task #22, after A)

Scoped 7/17 on the post-Slice-A census (base `bb6ee8d179c`, 276 phaseA): **46 distinct unregistered
residual paths**, saved to `/tmp/sliceB_residual_ranking.txt` (de-escape `\"`→`"` before counting).
The three walls that gate the hot dispatcher heads (innermost per record):

- `iter::adapters::map::Map::collect` (25 hits) → `setitem_slot`, `w_list_append`, `setitem_list`,
  `w_list_setitem`.
- `core::str::<Impl>::as_bytes` (7) → `getitem_slot`, `dict_entries_get_str`.
- `sync::atomic::AtomicBool::store` (15) → `object_setattr` (via `w_type_set_abstract`).

The 46 paths split into four orthodoxy buckets — **do NOT residualize a bucket-(N) core op** (that is
the non-orthodox band-aid: a silent perf regression no correctness test catches):

- **(F) foreign-opaque residual** — `Map::collect`, `Wtf8::*`, `IndexMap::{get,insert,get_index,
  with_capacity,get_index_mut}`, `AtomicBool::store`, `BigInt::{sign,to_u32}`, `fmt::rt::Argument::
  new_debug`. NOT auto-collected because the owners are EXTERNAL crates (`indexmap`/`wtf8`/`core`/`std`)
  → `iter_local_fns` (charon-reader:208) never sees them, and `collect_foreign_opaque_method_externals`
  (mir.rs:10681) only walks LOCAL opaque ADTs with a self-receiver and a modelable result
  (`foreign_opaque_method_result_valuetype` mir.rs:10773 declines `Option`/enum/tuple/ref). `BigInt`
  `sign`/`to_u32` owner IS opaque+local but the enum/`Option` result is declined. Fix = wrap the
  **pyre-side caller** in `#[dont_look_inside]` + `push_alias_pair`. Template already shipped:
  `w_dict_{store,lookup}_int_strategy` (jit_fnaddr.rs:557-580) residualize their internal
  `IndexMap::{insert,get}` wholesale.
- **(N) native-lowerable core op** — `core::slice::{get,first,index,as_ptr,chunks_exact}`,
  `from_raw_parts`, `f64::abs` (rtyper has `rtype_abs` rfloat.rs:216 → `float_abs`, method callsite
  unwired), `num::checked_div` (→ runtime-disc `Result`, template `try_lower_checked_neg` mir.rs:8610),
  `convert::{from,num::try_from}` (int-to-int `try_from` = the Slice-A-deferred `int_pow`/`pow`/
  `opcode_get_iter`), `RangeInclusive::new` (→ `rtype_builtin_range` rrange.rs:160), `Rev::next`,
  `mut_ptr::add`, `Vec::{index,index_mut}`. Real rtype/recognizer, never a residual.
- **(C) vec!/box/alloc cluster** — deferred to Slice C (capstone).
- **(P) pyre-internal accessor** — `set_async`, `EVAL_NESTING`, `EXC_CLASS_REGISTRY`,
  `subclass_range_read`, `GcType::type_id`, `Constants::{deref,index}`. `push_alias_pair` siblings of
  the jit_fnaddr.rs:697-758 runtime-state accessors.

**Sub-slice order (cheapest-per-leverage first; census depth LIES — head movement only when B1+B2
co-land):**

- **B1a (FIRST, done)** — the `str::as_bytes` / `Wtf8::as_str` identity-fold gap. Root cause:
  `is_string_as_bytes_identity` (mir.rs:7434) and `is_string_to_str_identity` (mir.rs:7405) both gate on
  `deref_impl_owner_leaf(fd)` matching an owner leaf, but `deref_impl_owner_leaf` (mir.rs:10530) resolves
  the owner through `resolve_impl_owner_adt_def_id_free`, which needs an ADT def-id. The primitive `str`
  is a `{Builtin:"Str"}` node with no def-id, so the literal `"str"` arm is dead; `Wtf8::as_str` fails a
  different way (`is_string_to_str_identity` gates owner `== "String"` only). Fix: gate both folds on the
  **receiver** being a string value via `tyref_is_string_value` (mir.rs:11539, which already handles
  `Builtin("Str")` + `String` + `Wtf8`/`Wtf8Buf` uniformly) rather than the impl-owner leaf. Do NOT
  touch `deref_impl_owner_leaf` (it also drives the `cast_pointer` thin-pointer rewrite).
- **B1b** — the (P) register-only accessors + `fmt::Argument::new_debug` fold into the existing fmt
  family (mir.rs:1757/7500).
- **B2** — the (F) foreign residuals (`Map::collect`, `AtomicBool::store`, `IndexMap` family, `BigInt`
  `sign`/`to_u32`). The str-dict path is STRUCTURALLY DIFFERENT from int/bytes: it uses the shared
  `dict_entries_get_str` helper (dictmultiobject.rs:179, not `#[dont_look_inside]`), not a dedicated
  strategy leaf — add a `w_dict_lookup_str_strategy`-style residual leaf or mark the shared helper.
- **B3 (LAST in B, real rtyper work)** — the (N) native lowerings.

### Slice C — vec!/NewList recognizer + repr-generic rtype_newlist (task #20, LAST, capstone)
- **Ca** Re-add the front/mir recognizer (verbatim from reverted `f41cb0496dc`): match
  `box_assume_init_into_vec_unsafe(box [e0..eN])` → `OpKind::NewList{args}`. Helpers
  `read_array_literal_elements` (mir.rs:13581) + `fmt_path_ends_with` (mir.rs:13673) still in tree.
- **Cb** `remove_dead_aggregates` (model.rs:2469) already sweeps the dead `Box::new_uninit`. No work.
- **Cc** repr-generic `rtype_newlist` (rlist.rs:395): accept BOTH `ListRepr` (Resized) AND
  `FixedSizeListRepr` (Fixed) — a never-mutated vec! annotates NON-resized → `FixedSizeListRepr`
  (rmodel.rs:3208). Fixed arm builds via `build_ll_newlist_helper_graph(ListLayout::Fixed)` +
  `build_ll_fixed_setitem_fast_helper_graph`. RPython-faithful (rlist.py:338-344 is repr-generic).
  **This code was written and verified to compile + pass `cargo test -p majit-translate`; it is
  correct but INSUFFICIENT alone — it only lifts graphs that fully rtype.** Recover it from this
  session's transcript / the reverted diff when Slice C lands.
- **Cd** Only after Slices A+B close every co-blocking wall does the recognizer become safe: with no
  vec!-graph dropping to legacy, `rtype_newlist` runs on every one, decomposing `NewList` to
  `ll_newlist` + `ll_fixed_setitem_fast` residual calls before assembly → no raw `newlist` reaches
  the default arm → `newlist/r>r` snapshot stays green. **Gate every Slice-C attempt on
  `default_bh_builder_unwired_set_matches_task_85_snapshot` (reads `insns.bin`, NOT census stderr).**

Verify each slice: `cargo test -p majit-translate` + census phaseA set-diff (count distinct
`[PREPASS phaseA fail]`) + `default_bh_builder_unwired_set_matches_task_85_snapshot` green +
`check.py` bit-exact 3-backend.

## Metric
Distinct `[PREPASS phaseA fail]` count in the census (268 on base `ccdc1a52be2`). Each slice measured
by set-diff. GOTCHA: the census **stderr** logs only phaseA *reasons* — the emitted `newlist/r>r`
opname lives in `insns.bin` (build OUT_DIR `target/debug/build/pyre-jit-trace-<hash>/out/insns.bin`;
`strings insns.bin | rg newlist`), not stderr. The cluster's hot graphs (`setitem_list`,
`getitem_list`, `w_list_setitem`, `setitem_bytearray`) only lift once ALL their stacked walls close —
expect most of the census movement on the LAST slice.
