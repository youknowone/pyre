# Walk-end commit contract — convergence roadmap

Status: **R1–R3 landed; R4–R6 open.** This is the authoritative plan for
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

---

## Open

### R4 — re-found the escape gate on a static effect class

Replace the runtime window comparison in `escape_opcode_window_clean`
(`residual_call.rs`; compares `frame_entry_count` and
`fbw_executed_effect_count` sampled at the first residual of the opcode) with a
per-callee effect classification decided once, the way `EffectInfo` is decided
at codewriter time (`jtransform.py:620-630`).

This is the item that makes `RewindProvenAtLatch` collapse into a real proof
rather than a named gap. Note the ordering hazard: re-founding the *existing*
gate on a commit-time sample instead (the obvious-looking move) would flip which
walks commit — the forcing residual bumps the odometer *after* the window is
sampled — and per TL;DR §3 the branch it would flip into is not safe either. So
this must be a static reclassification, not a resampling.

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

⚠️Do **not** drop the other conjunct, `!fbw_has_unjournaled_effect()`. Upstream's
`execute_and_record` executes *then* records (`pyjitpl.py:2647-2662`), so
"recorded symbolically but not yet executed" cannot arise there. pyre's walker
can record without executing, so that state is real and the conjunct is
load-bearing. It dies with the two-executor split, not with the odometer.

Blocker: per-level `PyFrame` materialization (`fbw_strict_fold_frame_reg`,
`vable_ops.rs`). Upstream avoids it only because `_nonstandard_virtualizable`
already degraded callee frames to heap fields (`pyjitpl.py:1120-1146`).
Tracked upstream of this as the inner-frame rebuild (gh#126/#215).

### R6 — vable write-through parity

Upstream's virtualizable is write-through: `_opimpl_setfield_vable` and
`_opimpl_setarrayitem_vable` both call `synchronize_virtualizable()`
(`pyjitpl.py:1194`, `:1246`). That is why `force_now` on `TOKEN_TRACING_RESCALL`
is a bare token store, commented "The values in the virtualizable are always
correct during tracing" (`rpython/jit/metainterp/virtualizable.py:248-255`).

pyre's shadow is explicitly lazy — "only generally valid at a merge point"
(`state.rs:4343-4357`) — and that laziness is the entire reason a mid-expression
operand-stack latch had to be invented. With write-through parity the live frame
is continuously authoritative, there is nothing to reconstruct at force time,
and the whole latch machinery retires: `flush_escape_state_with_latched_stack`,
`ESCAPE_OPCODE_WINDOW`, `EscapeFlushUndo`, `EscapeResumeKind`, and both
`RewindProvenAtLatch` and `RewindUnproven` with them.

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
