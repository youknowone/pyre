# Resized-list `append`/resize port (#131 value-type-method vein)

**Goal**: clear the `vec::Vec::push` phaseA cachedgraph-lift blocker (31 occ, concentrated in
`pop_n` 24 + `call_callable_with_mode` 4) by porting the resized `ListRepr`'s `append` + resize
machinery line-by-line from RPython. Honest baseline at start: **phaseA = 495**
(see memory `issue131-frontend-iternext-metric-neutral-registration-lever`).

## What already exists (do NOT re-port)
`majit/majit-translate/src/translator/rtyper/rlist.rs`:
- `ListRepr` (resized) struct + data shape `Ptr(GcStruct("list", ("length",Signed), ("items",Ptr(GcArray(ITEM)))))` — rlist.rs:596-646. ✅
- `rtype_len` → `build_ll_length_helper_graph` (getfield "length") — rlist.rs:669/804. ✅
- `rtype_getitem`/`rtype_setitem` via `list_rtype_getitem`/`list_rtype_setitem` with `ListLayout::Resized` — rlist.rs:701/724. ✅
- `build_ll_setitem_fast_helper_graph` (resized `ll_setitem_fast`: getfield "items" → setarrayitem) — rlist.rs:1048. **REUSE for append's final store.** ✅
- `rtype_method("reverse")` builder pattern to mirror — rlist.rs:744-766 + `build_ll_reverse_resized_helper_graph`. ✅
- Helper-graph op vocabulary confirmed present: `getfield`,`setfield`,`getarrayitem`,`setarrayitem`,`int_add`,`int_lt`,`conditional_call` (rlist.rs reverse/setitem builders). Multi-block + `conditional_call` example: the reverse loop builder ~rlist.rs:1116-1310.
- The 47 `rlist_runtime_deferred` free-fns (rlist.rs:329-403, incl `ll_append` 398, `ll_arraycopy` 378) are UNUSED stubs — the real impls are `build_*_helper_graph` graphs. Leave the stubs or delete as cleanup; they are not the wiring.

## RPython spec (the port source — cite these in code comments)
- `rpython/rtyper/rlist.py:185` `AbstractListRepr.rtype_method_append`:
  ```python
  def rtype_method_append(self, hop):
      v_lst, v_value = hop.inputargs(self, self.item_repr)
      hop.exception_cannot_occur()
      hop.gendirectcall(ll_append, v_lst, v_value)
  ```
- `rpython/rtyper/lltypesystem/rlist.py` `ll_append` (the resized one):
  ```python
  def ll_append(l, newitem):
      length = l.length
      _ll_list_resize_ge(l, length+1)
      l.ll_setitem_fast(length, newitem)
  ```
- `_ll_list_resize_ge` (lltypesystem/rlist.py:280):
  ```python
  def _ll_list_resize_ge(l, newsize):
      cond = len(l.items) < newsize
      ... jit.conditional_call(cond, _ll_list_resize_hint_really, l, newsize, True)
      l.length = newsize
  ```
- `_ll_list_resize_hint_really` (lltypesystem/rlist.py:200-239): overallocation growth
  (`some = 3 if newsize<9 else 6; some += newsize>>3; new_allocated = newsize+some`),
  `newitems = malloc(items.TO, new_allocated)`, `rgc.ll_arraycopy(items, newitems, 0, 0, p)`,
  `l.items = newitems`. The `newsize<=0` arm resets to the prebuilt-empty array.
- `ll_arraycopy` (`rpython/rlib/rgc.py`): element-wise copy loop (port as a helper graph:
  getarrayitem src[i] → setarrayitem dst[i], `int_lt` loop). Deferred stub at rlist.rs:378.

## ⚠️ PRIMARY GATE (confirmed) — this is an ANNOTATOR task FIRST, rtyper SECOND
The rtyper append/resize port below is **moot until the annotator models the receiver as
`SomeList(resized)`**. Confirmed at the annotator layer:
- `find_method` (unaryop.rs:~2634) only routes methods when `s_self` is `SomeValue::List(_)`,
  and its name map has NO `"push"` — only `append`/`extend`/`reverse`/`insert`/`remove`/`pop`/`index`.
- There is **no `Vec<T>`→SomeList modeling in the annotator** (rg empty). task#14 modeled the
  pyre-specific `FixedObjectArray`, NOT Rust's `std::vec::Vec`. So a `Vec<W_Root>` receiver does
  not annotate as `SomeValue::List`, so `push` never routes to `list_method_append`
  (unaryop.rs:1530, which does `listdef.resize()` + `generalize`), so the list is never marked
  resized, so the rtyper never mints a resized `ListRepr`.

**Therefore the epic ordering is:**
1. **Annotator slice (prerequisite, the foreign-value modeling vein — mirror task#14/#25):**
   model `Vec<T>` as `SomeList` and route its methods. Minimum for append: recognize the
   `Vec<W_Root>` receiver as `SomeList`, add `"push" => "list_method_append"` (and likely
   `"get"`/`Index`→getitem, `"len"`→len, `"truncate"`/`pop`, iteration) to the name map.
   **Scoping hazard (delicate, like task#14/#25):** Rust `Vec` is pervasive (scratch buffers,
   internal non-list collections). Modeling ALL `Vec<T>` as SomeList will mis-model non-list
   Vecs. Scope to the Vecs that semantically ARE RPython lists (e.g. `pop_n`'s arg-list build),
   or the population will mis-annotate broadly. Determine the discriminator before widening.
2. **Rtyper slice (this plan's steps below):** only reachable once step 1 makes `pop_n`'s Vec a
   resized `ListRepr`.

Verify step 1 landed by re-running the census and checking `pop_n` no longer emits
`vec::Vec::push` not-registered (it should change to a list-append op or a different blocker).

## Port steps (rtyper slice; each = one helper-graph builder, mirror `build_ll_reverse_resized_helper_graph`)
1. **`build_ll_arraycopy_helper_graph`** (`(src:Ptr(GcArray), dst:Ptr(GcArray), srcstart, dststart, length) -> Void`):
   multi-block loop `i=0; while i<length: dst[dststart+i]=src[srcstart+i]; i+=1`.
   (Or port `rgc.ll_arraycopy` faithfully.) Confirm the malloc/array opnames first (see Open Qs).
2. **`build_ll_list_resize_hint_really_helper_graph`** (`(l:Ptr(list), newsize, overallocate:Bool) -> Void`):
   the overallocation arithmetic + `malloc_varsize`(items array, new_allocated) + arraycopy(call #1) +
   `setfield(l,"items",newitems)`. Skip the `newsize<=0` empty-array reset initially if `pop_n` never
   appends to a 0-cap list (it newlists with capacity) — but port it for faithfulness.
3. **`build_ll_list_resize_ge_helper_graph`** (`(l:Ptr(list), newsize) -> Void`):
   `items = getfield(l,"items"); allocated = getarraysize(items); cond = int_lt(allocated, newsize);
   conditional_call(cond, ll_list_resize_hint_really, l, newsize, True); setfield(l,"length",newsize)`.
4. **`build_ll_append_helper_graph`** (`(l:Ptr(list), item:ITEM) -> Void`):
   `length = getfield(l,"length"); newsize = int_add(length,1); call ll_list_resize_ge(l,newsize);
   call ll_setitem_fast(l, length, item)` (reuse `build_ll_setitem_fast_helper_graph`).
5. **`rtype_method` arm**: add `"append" => { let (vlist,vitem)=hop.inputargs(vec![Repr(self),
   Repr(item_repr)])?; hop.exception_cannot_occur()?; helper=lowlevel_helper_function_with_builder(
   "ll_append", [ptr,item_ext], Void, build_ll_append_helper_graph); hop.gendirectcall(helper,[vlist,vitem]) }`
   — rlist.rs:744. Mirror reverse's builder closure capture of `ptr_lltype`/`item_lltype`.
6. **Route `vec::Vec::push`** → `rtype_method("append")`. `["vec","Vec","push"]` is NOT `core::*`, so
   `nonraising_core_bridge_opname` (flowspace_adapter.rs:637) does NOT fit (append RAISES via malloc).
   Use the `is_slice_reverse_segments` pattern (flowspace_adapter.rs:665 + translate_op routing to the
   getattr+simple_call method shape that reaches `rtype_method("append")`). Add `is_vec_push_segments`
   (`["vec","Vec","push"]`, argc==2) and route in translate_op alongside reverse. Confirm `op_canraise`
   classifies it raising (malloc can raise MemoryError) — UNLIKE reverse (cannot_occur). Check RPython:
   `rtype_method_append` calls `hop.exception_cannot_occur()` (rlist.py:188) → so append is NON-raising
   in RPython's model (the malloc OOM is not a Python-level exception). Mirror: route as non-raising.
7. **Verify the `pop_n` Vec annotates as a RESIZED `ListRepr`** (not FixedSizeList). If the annotator
   gives it FixedSizeList (non-resized), append won't apply — need `listdef.listitem.resized=True`
   propagation from the `Vec::push` mutation (RPython sets resized when append is seen). Check the
   annotator's listdef resized-flag logic; this may be the real gate.

## Verification
- Census (fresh build, pick stderr by `ls -t` mtime NOT size; build-script stderr at
  `target/debug/build/pyre-jit-trace-<hash>/stderr`):
  `rm -rf target/pyre-jit-trace-cache/pyre-jit-trace-codegen-cache-v1 && touch pyre/pyre-jit-trace/build.rs
   && PYRE_RTYPER_VERBOSE=1 PYRE_TWO_PHASE_RTYPE=1 cargo build -p pyre-jit-trace --no-default-features --features dynasm`.
  Success = `vec::Vec::push` drops out of the not-registered histogram AND phaseA < 495 (by however many
  of the 31 are sole-blocked vs multiply-blocked — expect partial, the rest co-block on bigint/bool::then/statics).
- `python3 ./pyre/check.py` — 158×2 corruption gate (list ops are exercised broadly; a mis-shaped
  append/resize WILL surface as crashes here).
- `cargo test -p majit-translate --features dynasm` — rlist builder unit tests.

## Open questions to resolve at implementation
- Exact opname for varsize array allocation in the helper-graph vocabulary (`malloc_varsize`?). Grep
  `rtype_malloc`/llmemory.rs/rbytearray.rs (`mn(size)=lltype.malloc(BYTEARRAY,size)`) for the spelling.
- `getarraysize` opname for `len(l.items)` (the FixedSizeList `rtype_len` uses it — copy from there).
- Whether `conditional_call` is directly emittable in a builder or needs the if-block expansion (the
  reverse loop builder shows the multi-block + branch pattern if conditional_call isn't a single op).
- The resized-flag annotator gate (step 7) — likely the true blocker; confirm before assuming the
  rtyper-side port alone suffices.

## ✅ IMPLEMENTED 2026-06-28 (append slice) — measurement-corrected scope
**Measurement correction** (the session's headline, consistent with the slice-iter correction):
the "31 vec::Vec::push" were **cachedgraph-LIFT** failures, NOT the phaseA-annotation metric. The
495 phaseA metric is composed ONLY of `annotate PANIC` (252) + `annotate Err` (243); `not registered`
is a SEPARATE later lift phase (34 distinct graphs). In the phaseA metric vec is only **8**, split
across 5 methods (push×3, as_ptr×2, resize_with, index, index_mut) — fragmented, partly unmodellable
(as_ptr). So the append port moves **lift coverage** (the #131 goal: rtyper handles more graphs →
legacy walker deletion), NOT the phaseA distinct-count proxy.

**What landed** (rlist.rs + flowspace_adapter.rs, NOT committed pending check.py):
- `ListRepr.rtype_method("append")` arm (rlist.rs) — `inputargs(self, item_repr)` +
  `exception_cannot_occur` + `gendirectcall(ll_append)`, minting sub-helpers in dependency order.
- `build_ll_append_helper_graph` (length + direct_call resize_ge + direct_call setitem_fast).
- `build_ll_list_resize_ge_helper_graph` (fused resize_ge+hint_really, grow-only specialization:
  `before_len < newsize` always so the `min`/`if before_len` guard collapses; malloc_varsize +
  direct_call arraycopy + setfield items; overallocation `some = (3 if newsize<9 else 6) + newsize>>3`).
- `build_ll_arraycopy_helper_graph` (rgc.ll_arraycopy specialized to start=0; 3-block loop).
- `is_vec_push_segments` + lift routing in flowspace_adapter translate_op: `vec::Vec::push(recv,item)`
  → `getattr(recv,"append") + simple_call(bound, item)` (mirror slice.reverse; Rust `push` → RPython
  list `append`). translate_op is shared annotation+lift, so it covers both phases.

**Measured result**: vec::Vec::push lift blockers **28→0**; **pop_n now fully lifts** (was sole-blocked
24); cachedgraph-lift distinct 34→33; NO helper-build errors (malloc_varsize/resize/arraycopy clean);
phaseA **495→495** (unchanged, expected — lift phase ≠ annotation metric). `call_callable_with_mode`
advanced to its next blocker `vec::Vec::extend_from_slice` (×4) — a follow-on (needs rtype_method
"extend" + ll_extend, NOT done). check.py corruption gate: PENDING.

Deferred (faithfulness convergence path): resize_hint_really `newsize<=0` reset + `overallocate=False`
branches (reached only by resize_le/resize, not append); the `if before_len:` guard (collapses for
grow-only). vec::Vec::index (4 lift) → getitem routing + extend_from_slice → extend: separate slices.

## Status / context
Tree clean on committed transitive-prune `08548262ce`. FOLLOW_EXIT iter_next fix + diagnostics are
stashed + `/tmp/followexit_plus_diag.patch` (metric-neutral, slice-iter not a current blocker).
bigint::to_f64/to_i64 (46) and bool::then (26) are SEPARATE epics (no rbigint repr; closure) — see
memory `issue131-valuetype-method-blocked-on-resized-listrepr-rbigint`.
