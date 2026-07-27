# Walk-end commit contract — convergence roadmap

Status: **R1–R4 landed; R5–R6 open.** This is the authoritative plan for
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

## Open

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

| | count |
|---|---|
| leg 3 `EntryCarrierCall` commits | 169 |
| leg 4 `CalleeRebuild` commits | 1 |
| refusal: callee is a generator | 151 |
| refusal: abort pc is not an exact segment anchor | 18 |
| refusal: first callee argument is not a `Ref` | 1 |

Row 1 never fired. Rows 4b and 6 never fired. Print the denominator: 149 of the
151 generator refusals come from one loop in `calls_closures.py`, and the 18
anchor refusals are spread over 5 files (`foriter_exempt_nested_foriter`,
`foriter_exempt_shared_generator`, `inline_subwalk_user_iterator`,
`selfrec_tail_exception_unwind`, `bridge_recursion_overflow`).

#### Row 4a is correct and must stay

Rebuilding a generator callee would run its body eagerly instead of producing a
generator object, so leg 4 refusing it is right. What is *not* right is that the
call was inlined at all: `resolve_inlinable_callee` (`jitcode_dispatch/mod.rs`)
has no `code_flags_make_generator` guard, so the walker inlines `gen(6)` and
starts walking the generator body, escaping only via an abort. Verified **not** a
miscompile (jit / `PYRE_NO_JIT=1` / CPython agree on a `yield`-per-`next` probe)
— it is wasted tracing, and a separate defect from this plan.

⇒ the honest denominator for R5 is **19**, not 170, and row 7's anchor is the
dominant real blocker.

#### Next slice: row 7 (the segment anchor)

`exact_floor_segment_anchor` (`jitcode_dispatch/mod.rs`) demands the abort
jitcode pc be the exact FIRST jitcode op of its Python pc's floor segment,
because leg 4 rebuilds a **Python** frame at a **Python** pc. Upstream needs no
such mapping: `_copy_data_from_miframe` (`blackhole.py:1711-1712`) resumes each
blackhole at its **jitcode** position. This is the F1 charter deviation
(py_pc ↔ jitcode), so the slice is scoped by it rather than by frame layout.

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
