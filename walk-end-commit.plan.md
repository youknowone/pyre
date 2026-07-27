# Walk-end commit contract — convergence roadmap

Status: **R1–R5 landed; R6 closed as refuted; R7 rejected.** The only
remaining rows carry named, measured causes and no corpus witness — see
"Residue" below. This is the authoritative plan for
retiring the walk-end commit gates in `run_perfn_walk`'s epilogue. It supersedes
the reasoning in the comments those gates carry, and it records two claims that
were in the tree and are **false**.

The contract itself (`WalkEndResume` in `pyre/pyre-jit-trace/src/trace.rs`) is
scaffolding that makes the deviation visible and typed. It is **not** the end
state. The end state is that most of it does not exist.

---

## TL;DR / disposition

1. **"Upstream never rewinds" is false.** `opimpl_str_guard_value`
   (`rpython/jit/metainterp/pyjitpl.py:1498-1511`) runs a real
   `do_residual_call` and then records `generate_guard(GUARD_VALUE,
   resumepc=orgpc)` — an *earlier* pc, which `capture_resumedata` stamps into
   the frame (`pyjitpl.py:2617-2620`). The sentence was in `trace.rs` and is now
   corrected there.

2. **The real upstream rule is static, not dynamic.** Upstream may rewind to an
   opcode start; it forbids rewinding past an *effectful* residual; and it
   decides **both** at codewriter time, never by measuring at runtime:
   - permission — `EffectInfo.EF_ELIDABLE_CANNOT_RAISE`
     (`rpython/jit/codewriter/jtransform.py:620-630`)
   - prohibition — per-opnum: the four guards that can follow a residual take
     `after_residual_call`, pinning the resume pc to the POST-call `self.pc`
     (`pyjitpl.py:2599-2602` → `194-198`)
   - upstream's only op counters are profiling-only and gate nothing
     (`rpython/jit/metainterp/jitprof.py:43-44`, `count_ops` is `pass`)

   So `FBW_EXECUTED_EFFECT_COUNT` discharges an obligation upstream **also
   has**, in a form upstream **does not use**. The convergence target is a
   static effect classification (R4), not deletion of the obligation.

3. **"Declining falls back to a safe replay" is false.** The store journal
   covers five folded classes only — list `STORE_SUBSCR`, `append`,
   append-promote, `IntMutableCell`, `sys_exc_value`
   (`pyre/pyre-jit-trace/src/jitcode_dispatch/mod.rs:4797-4874`) — and
   `FBW_FORITER_INFLIGHT` states outright that "the advance is an irreversible
   side effect with no journal undo" (`:4878-4880`). **Both branches of a
   decline can be unsound.** No new refusal may be justified by "the legacy
   replay is exactly-once", and this is why R7 is rejected.

4. **Leg 4 (`CalleeRebuild`) is the only orthodox leg**, and upstream's version
   of it is unconditional. Everything else converges toward it.

---

## The commit paths today

`pyre/pyre-jit-trace/src/trace.rs`, `WalkEndCommitLeg` / `WalkEndResume`.

| leg | resumes at | class | note |
|---|---|---|---|
| 1 `LoopHeader` | walk terminal | `Terminal` | |
| 2 `VableEscape` ×2 (blackhole adopts) | a frame RESULT | `Terminal` | |
| 2 `VableEscape` (epilogue, latched) | the escaping OPCODE | `RewindProvenAtLatch` | proof taken earlier, at the residual |
| 2 `VableEscape` (epilogue, merge-point fallback) | same pc, **no gate anywhere** | `RewindUnproven` → always declined | |
| 3 `EntryCarrierCall` | outer frame AT the CALL | `Rewind` | snapshot on `InlineAbortCarrier::Entry` |
| 4 `CalleeRebuild` | INSIDE rebuilt callee at abort pc | `AfterApplied` | latches BECAUSE the odometer moved |
| 5 `AbortPc` | the marker opcode | `Terminal` | |
| 6 `NestedInlineOuterCall` | outermost inline CALL | `Rewind` | snapshot on `FBW_ABORT_OUTER_RESUME` |
| 7 `BranchGuard` | abort pc + inflight LIFO | `Terminal` | |
| 8 `TerminateNoReplay` | no resume pc at all | `Terminal` | **not a flush leg**; keeps the journal by a different caller protocol and must not set `WALK_END_FLUSH_COMMITTED` |

`vstack_cur_pypc` is the pc the walk is **about to enter**:
`reconcile_vstack_at_boundary` reconciles the PREVIOUS opcode and then assigns
`new_pypc` (`jitcode_dispatch/vstack_mirror.rs:272-281`, `:518`). Both escape
flushes therefore take the same resume pc and both set
`last_instr = pc - 1` (`state.rs:4404`), so the escaping opcode re-runs either
way. They differ in operand-stack sourcing and in whether any gate ran.

---

## Landed

- **R1** — corrected the false "upstream never rewinds" claim in `trace.rs`, and
  the false "`vstack_cur_pypc` points one past the executing op" comment in
  `residual_call.rs`.
- **R2 (first half)** — settled `vstack_cur_pypc`; reclassified the latched
  escape flush from `Terminal` (which short-circuits the check) to
  `RewindProvenAtLatch`; deleted the dead `escape_opcode_window_effects`.
- **R3** — no deletion was performed, deliberately. The merge-point fallback's
  commit branch is already **unreachable at the code level** (`commit_walk_end`
  declines `RewindUnproven` unconditionally). Deleting more would break the
  escape-token force signal: the plain flush is what makes the live frame
  heap-authoritative for the escaping callee, and its undo can only run in the
  epilogue, after the callee has returned
  (`residual_call.rs`, `flush_active_frame_escape`).
- Naming the one journal-keeping path that sat outside the contract
  (`TerminateNoReplay`), so no commit path is anonymous in the census.
- **R4** — the escape gate is now founded on a declared effect class.
  `ESCAPE_OPCODE_WINDOW` holds `(py_pc, every_prior_residual_reentrant)`;
  `escape_opcode_window_clean` is a pure query; `escape_opcode_window_note`
  records each residual's `EffectInfo` class (`check_is_elidable` /
  `LoopInvariant`) after the call returns, so the gate and a force inside the
  callee both see the window as it stood *before* that residual. The ordering
  hazard below was respected — this is a reclassification, not a resample, and
  the CA-bench census is unchanged.

  `reentrant_residual` is deliberately stricter than `provably_side_effect_free`
  (it drops the `ForIterNext` exemption, which answers the in-flight FOR_ITER
  question, not the opcode-re-execution one), so it covers both runtime signals
  it replaced: advancing `frame_entry_count` or bumping the odometer both
  require a non-elidable, non-loop-invariant residual. The three journaled folds
  that also bump the odometer each terminate their own opcode, so no residual of
  the same opcode instance can follow one.

  Not addressed, and unchanged from before: the window is still keyed on `py_pc`
  alone, so an opcode revisited across walked inner-loop iterations still sees
  the first visit's verdict. That is conservative (decline → legacy); the
  refinement is a back-edge reset, and it belongs with R6.

---

## Residue

Nothing here is blocking. R5 landed, R6 is closed as refuted, R7 is rejected.
What is left is recorded so it is not re-derived, and each row states why it is
not being worked:

| row | state | why not worked |
|---|---|---|
| 3 `getarrayitem_vable_r` anchors | still refused | needs a color-liveness argument, not a measurement |
| depth-1 / N-frame (old assumed R5 blocker) | **unobserved, not refuted** | its gate is outside the named-refusal path, so the census cannot see it — no witness either way |
| `push_and_bump!` omits `jtransform.py:1898` | real gap | **no corpus witness**; prototyped, costs ~25% more vable stores, fixes nothing observed |
| legs 3/6 + store journals | keep | deleting them re-opens gh#467 double-apply (TL;DR §3) |

### R5 — generalize leg 4, then let legs 3 and 6 become unreachable

Upstream builds one blackhole per framestack frame, each at its own current pc
(`rpython/jit/metainterp/blackhole.py:1799-1821`, `:1711-1712`), and splices the
callee result into the caller **past** its call (`:1653-1662`). That is exactly
`AfterApplied`, and it is unconditional —
`run_blackhole_interp_to_cancel_tracing` ends `assert False  # ^^^ must raise`
(`pyjitpl.py:2956`). Upstream's answer to "give up inside an inlined callee" is
pop-callee, keep what ran, continue forward (`pyjitpl.py:1580-1600`); it never
rewinds the caller to its CALL.

Work: lift leg 4 from depth-1 / single-argument / no-closure to N frames, and
drop the `fbw_executed_effect_count() != executed_effects_before` conjunct,
which is a routing filter rather than a safety gate and currently makes the
orthodox path the exception. Legs 3 and 6 then retire by becoming
**unreachable**, not by deletion.

The eligibility set to lift is `inline_call.rs`, the `midbody_abort` payload
builder — in the order they refuse:

| # | refusal | upstream counterpart |
|---|---|---|
| 1 | `is_top_inline` (depth-1 only) | none — upstream loops over the whole `framestack` (`blackhole.py:1799-1821`) |
| 2 | ~~`fbw_executed_effect_count() != executed_effects_before`~~ | dropped (slice A) |
| 3 | `!fbw_has_unjournaled_effect()` | ⚠️KEEP — see below |
| 4a | callee is a generator / coroutine | ✅CORRECT, keep — see below |
| 4b | `cellvars` / `freevars` / non-null closure | none — upstream rebuilds any frame |
| 5 | `callee_arg_concretes.first()` must be a `Ref` (`x_arg`) | none — single-argument residue |
| 6 | every live stack slot a non-null `Ref`; every live local resolvable and not `Null`/`Bool` | partial — upstream reads typed registers per frame |
| 7 | `anchor_ok` / `abort_flush_call_jitcode_coord` / `depth`+`pcdep` resolvable | pyre's jitcode↔py_pc mapping, the F1 charter deviation |

#### ⛔The assumed blocker is not the measured one

The blocker recorded here before slice A — per-level `PyFrame` materialization,
i.e. rows 1/4/5 — **is refuted by measurement**. Slice A dropped row 2 and made
leg 4 preferred; the conversion was **zero**. Census over `pyre/bench/synth`
(315 files, `PYRE_FBW_DEBUG_ABORT=1`):

| | base | after slices A–E |
|---|---|---|
| leg 3 `EntryCarrierCall` commits | 169 | **3** |
| leg 4 `CalleeRebuild` commits | 1 | **17** |
| refusal: callee is a generator | 151 | 0 |
| refusal: abort pc is not an exact segment anchor | 18 | 3 |
| refusal: first callee argument is not a `Ref` | 1 | 0 |
| rebuild latched but declined at the flush | 0 | 0 |

Every remaining leg-3 commit is one of the 3 `getarrayitem_vable_r` anchors.

⚠️**Row 1 is NOT instrumented — do not read its silence as "never fired".** The
`is_top_inline && !fbw_has_unjournaled_effect()` gate (`inline_call.rs:2617`)
sits *outside* the closure whose `Err(&'static str)` arms are what print under
`PYRE_FBW_DEBUG_ABORT`, so a depth≥2 inline sub-walk abort (row 1) or an
unjournaled-effect abort (row 3) emits no line at all. Zero observations is
indistinguishable from not-measured. The only support is indirect and weaker
than it looks: the 3 surviving leg-3 commits are accounted for by the 3 anchor
refusals, which is *consistent* with row 1 not firing but does not establish it.
Rows 4b and 6 are likewise unobserved. Print the denominator: 149 of the
151 generator refusals came from one loop in `calls_closures.py`, and the 18
anchor refusals were spread over 5 files (`foriter_exempt_nested_foriter`,
`foriter_exempt_shared_generator`, `inline_subwalk_user_iterator`,
`selfrec_tail_exception_unwind`, `bridge_recursion_overflow`).

The orthodox leg is now the majority one.

#### ✅Row 4a is correct and must stay — the defect was upstream of it

Rebuilding a generator callee would run its body eagerly instead of producing a
generator object, so leg 4 refusing it is right. What was *not* right is that the
call was inlined at all: `resolve_inlinable_callee` (`jitcode_dispatch/mod.rs`)
had no `code_flags_make_generator` guard, so the walker inlined `gen(6)` and
started walking the generator body, escaping only via an abort. Verified **not** a
miscompile (jit / `PYRE_NO_JIT=1` / CPython agree on a `yield`-per-`next` probe)
— it was wasted tracing that discarded the whole trace 149 times.

Adding the guard (slice C) took leg 3 from 169 to **9** commits on its own, and
is what actually makes legs 3/6 rare rather than dominant.

#### ✅Slice B landed: row 7 (the segment anchor)

`exact_floor_segment_anchor` (`jitcode_dispatch/mod.rs`) demanded the abort
jitcode pc be the exact FIRST jitcode op of its Python pc's floor segment,
because leg 4 rebuilds a **Python** frame at a **Python** pc. Upstream needs no
such mapping: `_copy_data_from_miframe` (`blackhole.py:1711-1712`) resumes each
blackhole at its **jitcode** position — the F1 charter deviation (py_pc ↔
jitcode).

What the 18 refusals actually were, measured: the prefix between the segment
start and the abort pc is **only** `setarrayitem_vable_r` +
`setfield_vable_i` + `getarrayitem_vable_r`, every one of them aimed at the
callee's own `portal_frame_reg`. `portal_marker_first_jit_anchor` already
admitted a subset of exactly this (the `setfield_vable_i(VableField{index:0})`
prefix); it was generalized to `portal_vable_bookkeeping_anchor` and now serves
both abort kinds. Result: anchor refusals 18 → 3, leg 4 commits 1 → **11**,
leg 3 commits 169 → 159.

⚠️Two traps this slice walked into, both worth remembering:

- **`depth_for_jitcode_pc_pred` is keyed per Python pc**, not per jitcode pc
  ("Equals `depth_at_py_pc[python_pc_for_jitcode_pc(jit_pc)]`",
  `pyjitcode.rs`). Comparing it at the segment start and at the abort pc
  therefore *always* agrees and proves nothing. The real licence is that the
  rebuild rewrites locals, stack area, `valuestackdepth` and `last_instr`
  wholesale, so every vable write in the prefix is erased.
- The `setfield_vable_i` in the prefix writes `VableField{index:2}` =
  **`valuestackdepth`**, not `last_instr` (static field order is
  `last_instr, pycode, valuestackdepth, debugdata, lastblock, w_globals` —
  `virtualizable_gen.rs`). The pattern is `push_and_bump!`: store the slot,
  then bump the depth.

`getarrayitem_vable_r` is deliberately still refused (the remaining 3): it
writes a jitcode REGISTER, and the payload sources `live_stack`/`live_locals`
out of `concrete_registers_r` by color, so a clobbered color would be read back
as a live value. Lifting it needs a color-liveness argument, not a wider op set.

#### ✅Slice B also: the rebuild's fallback

Newly-latched rebuilds can still decline at the flush (5 of the 15 did), and
that decline used to land on the legacy replay — the unsound fallback of
TL;DR §3, and precisely the double-apply the entry carrier was built to close.
`MidBodyPayload` now carries `entry_fallback`, attached by
`fbw_set_abort_call_resume` under the entry latch's own zero-delta gate, and
the walk-end MidBody arm retries through the extracted
`try_commit_entry_carrier_call`. Commit totals are conserved: 169+1 → 159+11.

#### ✅Slice D landed: expression-position calls

All 5 rebuilds that latched and then declined failed one gate,
`can_flush_walk_end_state_after_outer_call`, and all 5 for the same reason: the
CALL was in **expression** position (static depth 7 at a 6-operand call, 2
after), while the preflight demanded `depths[post_call_py_pc] == 1` — the
statement-position specialization.

`outer_call_operands_below` now computes the residue
`depths[call_py_pc] - call_stack_len` and requires
`depths[post_call_py_pc] == below + 1`; the flush restores those operands under
the return value and sets `valuestackdepth` accordingly. `MidBodyPayload`
records only the call's own operand count, so the residue is sourced from
`entry_fallback.call_stack` — the entry carrier's
`reconstructed_all_ref_call_stack` is the caller's WHOLE operand stack at that
pc, slot-ordered from the stack base, so its prefix is exactly the residue.

⚠️GC: those refs are written **after** `frame.execute_frame` has run arbitrary
Python, so they are re-read from the live carrier (kept forwarded by the
abort-resume root area), never from the pre-execute clone.

#### ✅Slice E landed: the single-argument residue and the raising caller

Row 5 (`callee_arg_concretes.first()` must be a `Ref`) was **dead weight**.
`finish_for_call_with_globals_obj` binds `args` into the first `varnames` slots
and validates no arity; `try_commit_midbody_abort` then clears every one of
those slots to `PY_NULL` and rewrites them from `live_locals`. The seed never
survived, so `x_arg` is gone from `MidBodyPayload`, from its GC root visit, and
from the refusal list.

Removing it exposed the next one: `exception_delivery_stack_is_sourceable`
required `handler_depth == 0` — the same statement-position assumption slice D
removed from the return path, now on the raise path. `handle_exception` only
ever POPS down to the handler's recorded depth (`eval.rs`,
`pyopcode.py:151-173`), so restoring the `below` operands and setting
`valuestackdepth` from them serves any handler wanting at most that many. The
re-read of `below` from the live carrier is hoisted above the `execute_frame`
match so both arms use it.

#### Next slice

Leg 3 is down to 3 commits, all of them the `getarrayitem_vable_r` anchor.
Lifting it needs a color-liveness argument: the op writes a jitcode register and
the payload sources `live_stack`/`live_locals` out of `concrete_registers_r` by
color, so the case to prove is that the clobbered color is not one the payload
reads — not that the op is harmless. Row 1 (depth-1) and rows 4b/6 still have no
corpus witness; do not work them without one.

⚠️Do **not** drop the other conjunct, `!fbw_has_unjournaled_effect()`. Upstream's
`execute_and_record` executes *then* records (`pyjitpl.py:2647-2662`), so
"recorded symbolically but not yet executed" cannot arise there. pyre's walker
can record without executing, so that state is real and the conjunct is
load-bearing. It dies with the two-executor split, not with the odometer.

Per-level `PyFrame` materialization (`fbw_strict_fold_frame_reg`,
`vable_ops.rs`; the inner-frame rebuild, gh#126/#215) is still the blocker for
row 1 *in principle* — upstream avoids it only because
`_nonstandard_virtualizable` already degraded callee frames to heap fields
(`pyjitpl.py:1120-1146`) — but no corpus case reaches it yet, so it is not what
gates progress today.

### R6 — CLOSED: the latch is orthodox; nothing is missing on this path

⛔**R6 is not a work item. Two successive premises were recorded here and both
are refuted by measurement.** Keep this section so the theory is not re-derived.

**Premise 1 (refuted): "pyre's shadow is lazy where upstream writes through."**
`TraceCtx::vable_setfield` and `vable_setarrayitem_indexed` both end in
`self.synchronize_virtualizable()` (`trace_ctx.rs`), mirroring `pyjitpl.py:1194`
/ `:1246`, and `write_boxes` parity exists. The fold path that writes nothing
(`vable_ops.rs`) is gated on `fbw_strict_fold_frame_reg` =
`callee_shadow.fold_frame_reg`, set only for an inline **callee** sub-walk,
while the escape latch is gated on `!inline_subwalk` — a different path.

**Premise 2 (refuted): "the missing write is the `pushvalue` array store."**
The latch was instrumented to compare every resolved slot against the shadow
(`residual_call.rs`, `[r6-latch]`, gated on `PYRE_FBW_DEBUG_ABORT`) and censused
over `pyre/bench/synth`:

| | |
|---|---|
| disagreements | **10**, in 2 files (`getframe_stored_fback_walk.py` 5, `getframe_force_cancel_journal.py` 5) |
| slot index | **`rel == len-1` in 10/10** — always the in-progress opcode's TOS |
| shadow value | **`Ref(GcRef(0))` (NULL) in 10/10** — never a stale non-null |
| shadow **box** | **`ConstPtr(GcRef(0))` in 10/10** — a compile-time NULL *constant* |

That last row is what settles it. The slot does not hold *no* write; it holds a
**deliberate NULL constant**, which is exactly what `emit_popvalue_ref!` emits:
`pyframe.py:411-417 popvalue_maybe_none` → `setarrayitem_vable_r(
locals_cells_stack_w, depth, ConstPtr.NULL)` via `jtransform.py:1898`. The
in-flight opcode had already **popped** that slot before its residual forced.

So the vable array is *correct*, and upstream agrees with it — RPython's
`popvalue` NULLs `locals_cells_stack_w[depth]` the same way. What the latch
holds is the operand the in-flight opcode already consumed, which is precisely
what upstream keeps in `MIFrame.registers_r` and what
`convert_and_run_from_pyjitpl` resumes a blackhole from. **The latch is the
orthodox register-level mirror, not a workaround.** It does not retire, and
neither do `ESCAPE_OPCODE_WINDOW`, `EscapeFlushUndo`, `EscapeResumeKind`,
`RewindProvenAtLatch` or `RewindUnproven` on this argument.

Directly disproving premise 2: `LOAD_ATTR` — the escaping opcode in both witness
files — *already* emits the push mirror via `emit_pushvalue_ref!`
(`codewriter.rs`), and the slot still reads NULL. Emitting the store is
demonstrably not sufficient, because the pop follows it.

#### Witness-less residue: `push_and_bump!` omits the `:1898` half

Genuine and separate from the above. `pyframe.pushvalue` is two writes —
`locals_cells_stack_w[depth] = w_object` (`jtransform.py:1898`
`do_fixed_list_setitem`) and `valuestackdepth = depth + 1` (`:844`).
`emit_pushvalue_ref!` and `emit_load_fast_ref` emit both; the generic
`push_and_bump!`, taken by every residual-call and HLOp result (38 sites),
emits only `emit_vsd!`. A guard-failure resume at an opcode **boundary** just
after such a push would therefore read a NULL stack slot.

Measured cost of closing it (prototyped, then reverted): adding the store to
`push_and_bump!` under a `Kind::Ref` gate takes
`getframe_stored_fback_walk.py` from **140 to 175** emitted
`setarrayitem_vable_r`, ~25% more vable stores, and **changes none of the 10
disagreements** — it is not a fix for anything observed.

⛔**Do not land it without a corpus witness.** Per this file's own evidence
rules, a row with no witness is not worked. To get one, look for a guard that
fails at an opcode boundary immediately after a residual-call push and observe
whether the resumed frame sees NULL at that slot.

### R7 — REJECTED: do not delete legs 3/6 or the journals now

Re-opens the gh#467 double-apply for callees that cannot be resolved up front,
and the fallback they would retreat to is unsound for the effect classes the
journal does not cover (TL;DR §3).

---

## Evidence rules for this area

- **"Zero corpus reaches" is evidence about the corpus, not the path.** The
  merge-point fallback measured 0/334 on both backends, but it fires only when
  the latch *fails* (non-`Ref`/null slot, `vstack_valid == false`, a bridge
  trace, an inline sub-walk), and the corpus structurally cannot produce those.
  The defensible licence for refusing it is the code-level one: `commit_walk_end`
  declines `RewindUnproven` unconditionally. State it that way.
- `ESCAPE_PLAIN_FALLBACK` bumps only under `!latched && flushed`, so a flush
  that *declines* is never counted; the counter says nothing about decline
  frequency.
- Always print a reachability denominator next to a hazard count.
- `timeout(1)` does not exist on this macOS; use `perl -e 'alarm N; exec @ARGV'`.
  A corpus loop using `timeout` runs zero benches and every probe reads 0.
- Walk-level observability is `fbw_diag` (`trace.rs`) plus the `pyre_fbw_diag`
  **export** and the runner decoder. It must stay an export: adding a host
  *import* shifts wasm function indices and breaks JIT-baked ones.
