# Closure roadmap: flatten-graph residual + Path 4 epic (#238) + pc_map (#73), PyPy-faithful

Status snapshot (2026-07-02, branch `flatten-graph`, HEAD `3c5191088ea` atop issue62 P2
framestack commits). Every phase below is sliced so each slice lands gate-green
(cargo test + check.py ×2 backends + 599-prog corpus sweep + adversarial refute WF for
resume-zone slices). One green slice per session; never push.

## Ground truth (censused this session)

Runtime flat-map (`stack_slot_color_map`) readers remaining:

| reader | site | gate | class |
|---|---|---|---|
| hazard flat fallback | jitcode_dispatch.rs:9034 (`kept_stack_has_boxed_int_hazard`) | `!mirror_covers_kept` | bounded, DEAD on CI, 22 safe-smallint fires on 9 adv progs |
| overlay legacy leg | jitcode_dispatch.rs:9644 (`stack_sync`) | mirror-primary/legacy-fallback per slot | same trigger population |
| encoder snapshot | trace_opcode.rs:1599, 1666 (`emit_live_refs`, via `semantic_ref_slot_for_reg_color`) | none | color→slot inversion, PyPy never inverts |
| bridge maps | state.rs:1565 (`bridge_semantic_maps_at`) | none | bridge resume semantic translation |
| inline recipe | `reconstruct_inline_recipe` (state.rs) | vable-array branch reconstructs mid-body resume (no decline); register-section `color==semantic` gate DEAD corpus-wide (see S4c RESOLVED 2026-07-10) | migration already complete; gate relaxation inert — DO-NOT-RE-ATTEMPT |
| producer | codewriter.rs:11317-11321/11502/11656; canonical_bridge.rs:207-235 (identity) | — | — |
| plumbing | pyjitcode.rs:195 field, state.rs:1403 accessor, fixtures | — | — |

Path 4 (#238) residual surface: `walker_slot_for_variable` (~15 sites) +
`pair_walker_slot`/`_if_absent` seeding (portal args + block inputargs),
`filter_cross_slot_coalesce_pairs` (codewriter.rs:11048), `_with_pairs` regalloc
variants (regalloc.rs:474/500/696/713), `pin!` 77 sites (stack/exc only; body locals
already retired by #347), `semantic_ref_slot_for_reg_color` decoder inversion
(state.rs:1662 + 4 dispatch sites), `local_slot_color_map`, and the decode color
overlay in `restore_guard_failure_values` (state.rs:8404) — the
PYRE_VABLE_DECODE_SKIP_COLOR gate from the 2026-06-11 slice is no longer in the tree
(rebased away), so the overlay-redundancy claim (731 decodes, 0 diff) needs
revalidation before deletion.

pc_map (#73): dense py-pc→jitcode-offset map, two consumer classes — resume entry
points (sparse-safe per #296) and guard-liveness lookups at arbitrary tracing PCs
(`get_list_of_active_boxes` → carry-forward via `derive_pc_live_indices_from_sparse`,
codewriter.rs:773/11198). #209 dense→sparse migration REFUTED — do not retry; the
faithful endpoint is keying everything by jitcode pc (blackhole.py:1712 model).

Root blocker (characterized 2026-07-02): the 22 residual mirror holes are
short-circuit merge-TOS values. `PopJumpIfTrue` is a pop, not a producer;
`PopOnlyOrSideStore` reconcile truncates with no box for the kept TOS; the TOS is the
merge of two edges the single-pass structural walk didn't traverse. In RPython flow
graphs this value is NOT a phi problem: **merges pass values as Link args into the
target Block's inputargs** — the merge value is simply the merge block's inputarg
Variable. pcdep fails to name it only because `pcdep_slot_var_resume`/`live_oracle`
never records block-inputarg Variables at block-entry resume PCs. That is the
keystone fix.

## Phase 1 — keystone: pcdep totality at merge PCs (task #355)

PyPy anchor: flowspace Block.inputargs + codewriter/flatten.py `insert_renamings`
(link args → target inputargs); liveness.py per-instruction live sets include the
inputargs live at block entry. The flat coloring already carries the unified color
for the merge value (link-arg coalescing); pcdep just has to expose it per PC.

### REFUTED sub-approach (2026-07-02, codex+verify): block-entry inputarg seeding
The first attempt seeded `pcdep_slot_var_resume` at block-entry PCs (`py_pc ==
start_pc`) from the target block's Ref-kind inputargs (helper
`seed_block_entry_ref_stack_inputargs`). Built green, check.py 169×2 — because it is
**INERT for the goal, with an unfaithful side-effect**. Census evidence (env-gated
`PYRE_P1_CENSUS` at the hazard reader jitcode_dispatch.rs:9002/9038 + `PYRE_P1_SEED_CENSUS`
in the producer), e41 + 8 siblings:
- The hazard flat-fallback fires at resume **py=207** (a MID-block guard inside the
  except handler), slot 4. Total across the 9 progs = **exactly 22** flat fires,
  **unchanged** by the seed; pcdep-hit = **0** everywhere.
- The seed only ever runs at block-ENTRY PCs (e41: py=103, py=219) — never at py=207.
  So it adds nothing the residual guard consults.
- Worse, at py=219 it **overwrote** already-correct walker-stack pcdep entries
  (vars 3031/3042/3035 → block inputargs 1644/1645/1646) at slots 2/3/4 — a semantic
  change at a non-residual PC with no proven benefit; green only because those PCs are
  not exercised as residual resume points.
REVERTED (working tree clean, nothing committed). Lesson: the residual value is a
**kept operand-stack temp live across a mid-block guard**, not a block inputarg at a
different PC. And it is NOT a Variable in the walker's symbolic stack at the resume PC
— otherwise the existing stack loop at codewriter.rs:6993 (`current_state.stack` →
`pcdep_slot_var_resume`) would already name it and pcdep-hit would fire. It fires 0,
so at py=207 the walker's `current_state.stack[residual_slot]` holds a non-Variable
(Constant / NONE-sentinel / short stack). THAT is the real gap.

### ★ Phase 1 STATUS (2026-07-02): S1a DONE, S1b LANDED `d388c8fe6a6`, residual = 0
S1a census (PYRE_VSTACK_DIAG + push-chokepoint probe, all reverted) classified the
22 residual slots: [EDGE-COPY] bucket EMPTY (recovered_hit=false everywhere);
[MERGE] bucket EMPTY; ALL 22 were one root cause — **LoadConst's pushvalue lowering
never recorded the `setarrayitem_vable_r(_, ConstInt(stack_slot), const)` stack
write** (jtransform.py:1898 emits it for every pushvalue; pyre's PY_NULL pushvalue
and StoreFastLoadFast LOAD-half already had it). The mirror hole was the direct
consequence: no vable store → no `vstack_last_ref` → reconcile ResultToTos left TOS
NONE → truncate carried NONE into the merge resume pc. S1b = 29-line codewriter fix
in the `Instruction::LoadConst` arm. Verified: flat-fallback fires 22→0 on the 9
progs; check.py 169×2; cargo test pyre-jit 284/0 + pyre-jit-trace 260/0; 9 progs
byte-exact vs CPython; soak_exc 60/60. The earlier "SSA-phi / non-walked-edge merge"
theory is REFUTED — the walked path itself simply never recorded the const push.
NEXT = S1c broader corpus census (599) then Phase 3 reader deletion.

### ★ Phase 1 S1c (2026-07-03): census 0 corpus-wide + THREE mirror-correctness fixes

S1c census: 286 surviving corpus progs (313 of 599 lost to /tmp cleanup) → flat-fallback
fires = **0**. Adversarial refute WF (6 shape families × 8-10 hostile progs, 63 total) →
flat fires 0, BUT **8 JIT≠CPython divergences**. Bisect against pre-S1b split them:
- m22/m06_g "regressions" were NOT S1b bugs: their big consts previously hit the
  boxed-int hazard decline (via `const_ref_slots_at_pc`) which blocked compilation and
  MASKED the real defect (beneficial-cutoff pattern, cf #418). Small-const variants
  (m06_n/m08_a) crashed identically WITHOUT S1b.
- Underlying defects (all found via PYRE_VSTACK_DIAG on the 16-line m06_n repro), all
  fixed this slice in jitcode_dispatch.rs:
  1. **Cross-arm layout boundary graft**: the FBW walk visits jitcode blocks in LAYOUT
     order, so after walking one branch arm the next boundary can enter the OTHER arm
     (`22→20` in m06_n). `reconcile_vstack_at_boundary` applied the prev op's
     ResultToTos there, grafting the else-arm's const into the taken-arm entry TOS =
     the CALL null-marker slot → deopt restored a real object there → "f() takes 2
     positional arguments but 3 were given". FIX: `vstack_fallthrough_reaches`
     sequential gate (trivia-skip + bounded unconditional-jump follow, liveness.rs
     delta arithmetic); non-sequential boundary = resize + shadow-reseed only.
  2. **Overlay legacy override of shadow-correct slots**: for a NONE mirror hole in a
     VALID mirror, `stack_sync` fell back to the legacy
     `registers_r[stack_slot_color_map[s]]` read, which OVERRODE the shadow's already-
     correct write-through value (the null marker) with a stale color's box. FIX: a
     valid mirror is the SOLE overlay source (hole → leave shadow); legacy read only
     for invalid-mirror walks. This is the P3 overlay-leg semantic, landed early
     because it was load-bearing for correctness.
  3. **Method-form LOAD_ATTR misclassified ResultToTos**: `o.m(...)` pops obj and
     pushes `[attr, self]` (two slots change); single-TOS modeling left the popped
     receiver STALE in the attr slot → "'object' object is not callable". FIX:
     `is_method()` form → MultiResultFromShadow (clears both slots, shadow reseed).
- Result: 5/6 minimized repros fixed INCLUDING pre-existing silent-wrong m11
  (condexpr-sum acc duplication, a live #225-family default-path miscompile) and
  m06_g/m06_n/m08_a/m22. Refute corpus rerun: 110/118 pass; all 8 fails = ONE
  pre-existing exc bug (m7: raise inside callee try/finally caught by caller except →
  "exception must derive from BaseException" after warmup; NOT vstack; #359-family
  sibling, recorded for P8).
- S1b VINDICATED (not reverted): the exposure was decline-unmasking of a pre-existing
  resume defect, and S1b's shadow completeness is a precondition for fixes 1-2.
- P3 note: the hazard's flat-consult branch now models a source the overlay no longer
  uses under a valid mirror (semantically stale, measured 0 fires) → delete in P3
  along with the invalid-mirror-only legacy leg residue.

### REVISED Phase 1 (post-refutation): trace-time capture parity — complete the mirror

Re-framing on PyPy parity grounds. PyPy has NO deopt-time reconstruction from static
maps at all: guards capture resume data at **trace time** from the MIFrame registers /
`virtualizable_boxes` (`capture_resumedata`; pyjitpl.py `get_list_of_active_boxes`),
and those registers hold EVERY live operand-stack cell, of every kind — i/r/f banks,
constants recorded as ConstInt. The pyre analog of that layer is the vstack mirror
(`ctx.vstack_boxes`, production-default kept-stack source since the symstack S7 flip).
Therefore the faithful keystone is NOT pcdep totality (a decode-side map — refuted
above as the wrong layer for this residual); it is **mirror totality at kept-stack
branch guards**. pcdep keeps its role on the decode paths (Phase 4); its merge-PC gap
stops mattering for the hazard/overlay once the mirror covers the slot. Old Phase 2
merges into this phase.

- S1a decision probe (revert before landing): classify each of the 22 residual slots
  by WHY the mirror lacks a box at the guard. Instruments: the overlay diag
  (`PYRE_VSTACK_DIAG`, extended with `recovered_hit` + raw register read) + the
  reconcile trace. Buckets:
  - [CONST/INT-BANK] the value was pushed as a constant / Int-bank temp the Ref-only
    mirror never records (comment at the overlay: "an Int-bank stack temp the
    Ref-only mirror does not hold"). PyPy holds these in registers_i / ConstInt.
  - [EDGE-COPY] the value is already recovered by the #420 branch-handler decode of
    the not-taken edge's `ref_copy` moves (`BRANCH_GUARD_KEPT_RECOVERED`), but only
    as a COLOR-keyed patch on the legacy flat read — the mirror itself stays NONE.
  - [MERGE] genuinely produced on a non-walked edge with no local recovery (the
    residual "merge hole" proper).
- S1b by bucket, each a separate gate-green slice, in this order:
  - [EDGE-COPY] → make #420 recovery mirror-native: seed the recovered values into
    `vstack_boxes` by SLOT at the guard (the branch handler knows the edge moves), so
    `mirror_covers_kept` turns true and the color-keyed legacy patch retires. This
    removes a flat-map dependency by itself.
  - [CONST/INT-BANK] → extend the mirror to record constant pushes / Int-bank cells
    (registers-hold-all-kinds parity; snapshot treats a small-int const as trivially
    restorable ConstInt).
  - [MERGE] → reconcile-level fill across the `PopJumpIf*` boundary from the edge's
    known move set (dataflow-aware reconcile, bounded to the branch shape) — only if
    this bucket is nonempty after the first two land.
- S1c verify + flip: census shows hazard flat-fallback AND overlay legacy-leg fires →
  0 across the 9 progs and the 599 corpus; check.py ×2 backends; adversarial refute
  WF over short-circuit/chained-comparison/exc-handler shapes; byte-exact on the 9.

Exit criterion: `mirror_covers_kept` true at every kept-stack branch guard in the
corpus — the two bounded flat readers become empirically dead everywhere, unblocking
their deletion (Phase 3). Est. 2–3 sessions.

Ultimate-parity note: even the mirror-slot bookkeeping is transitional. The #73
endgame (Phase 7) resumes at the jitcode pc with positional vable restore
(blackhole.py:1712), after which no py-slot map of any kind survives.

## Phase 2 — MERGED INTO PHASE 1 (revised)

The old "pcdep names the merge value, then the mirror fills from pcdep" ordering is
obsolete: the mirror is completed directly at the capture layer (Phase 1 revised).
The #370 coverage-cascade items that are NOT kept-stack-guard residuals (FOR_ITER /
UNPACK / inline-subwalk / exc paths) remain as opportunistic slices here, each
deleting one more mirror-miss decline gate-green. PyPy anchor unchanged:
MIFrame.registers_r is always complete; `opimpl_goto_if_not` never declines.
Est. 0–2 sessions depending on what Phase 1's census leaves over.

## Phase 3 — delete the two bounded resume readers

Delete the hazard flat fallback (jitcode_dispatch.rs:9034 block) and the overlay
legacy leg (9644 capture + legacy read loop; mirror becomes the sole stack_sync
source). Bar: census 0 fires, check.py ×2, corpus byte-exact, refute WF. This closes
the flatten-graph branch's residual issue proper. Est. 1 session.

## Phase 4 — migrate compile-path flat readers to pcdep

- S4a encoder: trace_opcode.rs:1599/1666 stop inverting colors via
  `semantic_ref_slot_for_reg_color` + flat map; read pcdep (per-PC) instead. PyPy
  anchor: pyjitpl.py:194 `get_list_of_active_boxes` reads MIFrame registers by
  jitcode liveness — no inversion exists upstream.
- S4b bridge: `bridge_semantic_maps_at` (state.rs:1565) → pcdep-driven.
- S4c inline recipe: `reconstruct_inline_recipe` (state.rs) — its decline comment
  literally asks for "a per-pc color→semantic map the metadata does not carry";
  after Phase 1 pcdep IS that map. Replace the
  color==semantic identity check with a pcdep lookup → mid-body inline-callee resume
  stops declining. This is the #124 core; it unblocks #329 (mutual-recursion deopt)
  and #339 (return-in-finally). Est. 2–4 sessions.

### ★ S4c RESOLVED (2026-07-10, census on current HEAD): recipe and framestack are
### COMPLEMENTARY (decode→compile); recipe migration ALREADY complete; the
### color==semantic gate is DEAD. No code slice — evidence recorded.

Re-grounding this session found the roadmap above stale by ~8 days. On current HEAD
the two coordinated paths are NOT competing alternatives — they are sequential:
- `reconstruct_inline_recipe` is the DECODE side. On a guard failure with
  `resume_data.frames.len() > 1` it decodes each inlined-callee frame into a
  `ReconstructRecipe`; those recipes populate the `BridgeInlineCarrier`. If ANY
  callee declines, the carrier's recipe list is cleared.
- `drive_bridge_framestack_walk` (trace.rs) is the COMPILE side. It CONSUMES the
  carrier's recipes, reconstructs the callee framestack, and walks it forward to
  compile the cross-frame bridge (depth-1; `recipes.len() != 1` and any setup
  failure fall to `SafeAbortReconstruction`).
The framestack walk therefore DEPENDS on the recipe and cannot supersede it. The two
gates it was originally behind (`PYRE_P2_FRAMESTACK` / `PYRE_P2_FS_COMPILE`) were
already removed on a later commit; the depth-1 walk is production-default, with the
`SafeAbortReconstruction` safe-abort retained. **Decision: MIGRATE (keep) the recipe;
do not retire it.**

The migration is ALSO already functionally complete. Since #327 (`2e1a77f8a9e`) the
recipe has TWO reconstruction branches:
1. the vable-array (portal-shaped virtualizable callee) branch — reads locals by
   SEMANTIC slot directly from the reconstructed frame's `locals_cells_stack_w`
   vable array. It has NO color==semantic decline and reconstructs mid-body resume
   without declining.
2. the register-section branch — carries the pcdep color→slot inversion AND, in
   front of it, the leftover `color==semantic` identity gate.

Census (env-gated probes at every recipe decline site + both branch entries;
reverted before commit) over ~170 programs (12 bench + 15 adversarial mid-body
inline-callee deopt shapes + the 155-prog synth corpus), dynasm backend, **4264
recipe reconstructions**:
- **4257 (99.8%) took the vable-array branch and succeeded.**
- **7 (0.16%) declined via the int/float-bank gate** (a callee with a live int/float
  register — no boxed-Ref source; falls back to the single-frame bridge, program
  still correct via blackhole).
- **0 ever reached the register-section branch**, so the `color==semantic` gate
  fires **zero** times and its pcdep inversion is dead-equivalent-to-identity.
Closures/generators are declined earlier (the freevars/cells gate), which is why the
register-section shape is architecturally unreachable in the current
virtualizable-every-frame model.

Consequence: the #124 core ("mid-body inline-callee resume stops declining") is
ALREADY satisfied by the vable-array branch — fib_recursive (mid-body branch guard
flips as n crosses 2), inline_helper, nested/mutual recursion, exceptions-in-callee,
kwargs calls, ref-typed locals all reconstruct and compile byte-exact vs CPython.
Relaxing the `color==semantic` gate (the literal S4c edit) is **INERT** and
**unexercisable**: it changes nothing on the whole corpus and cannot be verified on
the branch it targets. Per the load-bearing-decline discipline (#418) and the refuted
inert-seed precedent (Phase 1 block-entry seed), an inert relaxation of an
unexercisable decline in this miscompile-prone zone is NOT landed. DO-NOT-RE-ATTEMPT
the gate relaxation; if the register-section branch is ever shown reachable, migrate
it THEN with runtime coverage in hand.

Remaining genuine (but rare / risky) recipe work, if pursued later: the 7
int/float-bank declines — box the unboxed int/float local into a Ref for the
reconstructed frame so the multi-frame bridge compiles instead of falling back. This
is a perf optimization (the fallback is already correct), lives in the resume
miscompile zone, and needs its own adversarial WF.

Adversarial-WF finding (2026-07-10, PRE-EXISTING, code pristine): #339 return-in-
finally SILENTLY MISCOMPILES on the cranelift backend. Shape: a callee wrapping its
`return` in `try/…/finally` that is called twice and summed (`f(x) + f(x+1)`) prints
NOTHING (empty stdout, exit 0) on cranelift while dynasm + CPython print the correct
sum. Isolated triggers pass: no-finally passes; finally with a single call passes —
the miscompile needs finally-wrapped-return AND two summed inlined calls. dynasm is
correct, so it is a cranelift-specific resume/finally lowering bug, not a recipe
defect. Repro saved off-tree. NOT the P4c slice; recorded for Phase 8 (#339).

## Phase 5 — delete `stack_slot_color_map` (and then `local_slot_color_map`)

When Phases 3–4 leave zero runtime readers: delete the field (pyjitcode.rs:195),
producer loop (codewriter.rs:11317-11321/11502/11656), the canonical_bridge identity
producer (portal bridges get identity semantics from pcdep-absent convention instead),
accessor (state.rs:1403), fixtures/docs. Then repeat the same reader-migration →
deletion for `local_slot_color_map` once the decoder overlay is pcdep-total (Phase 6).
This closes #267/#363 fully. Est. 1–2 sessions.

## Phase 6 — Path 4 epic (#238) endgame

The 2026-06-11 audit stands: resume.py virtualizable_boxes is ported end-to-end; the
sole decode deviation is the color→slot overlay. The slot-stability machinery
(walker slot pairing, cross-slot filter, `_with_pairs`, stack/exc `pin!`) exists ONLY
to keep colors slot-stable for the flat-map contract; pcdep totality dissolves that
need. Order matters — decode first, then regalloc:

- S6a decode overlay: re-add the equivalence probe on current HEAD (the old
  PYRE_VABLE_DECODE_SKIP_COLOR slice was rebased away), revalidate the 0-diff claim
  (previously 731 decodes, 0 diff; blocker #326 is resolved), then delete the overlay
  in `restore_guard_failure_values` and the decoder's
  `semantic_ref_slot_for_reg_color` uses (state.rs:1662 + jitcode_dispatch
  6479/6523/6572/6617) — decode becomes positional/pcdep like resume.py.
- S6b regalloc: retire `pair_walker_slot`/`pair_walker_slot_if_absent` seeding and
  `walker_slot_for_variable`, `filter_cross_slot_coalesce_pairs`, and the
  `_with_pairs` regalloc variants → plain `perform_register_allocation`
  (regalloc.py:98-112 parity, which the orthodox `try_coalesce` already matches).
  Each retirement is a separate slice with adversarial WF (this is the resume
  corruption zone).
- S6c pins: retire the stack/exc `pin!` sites (77) the same way — they are
  slot-stability pins; body-local pins are already gone (#347).
- S6d `local_slot_color_map` deletion (joins Phase 5's second half).

Note the perf gate memory: a previous coalescing flip attempt hit a cranelift timeout
(codewriter.rs note) — carry a perf bar (fannkuch/spectral/nbody wall-clock) alongside
correctness on every S6b slice. Est. 3–5 sessions.

## Phase 7 — #73 pc_map retirement (the dominant-goal endpoint)

Faithful model: resume data carries the jitcode pc directly (blackhole.py:1712);
liveness is keyed by jitcode position (codewriter/liveness.py, jitcode.py); no
py-pc→jitcode map exists upstream. #209 (dense→sparse) is REFUTED — the guard-liveness
consumer class needs carry-forward semantics the sparse map can't give; so migrate
consumers off py-pc keying instead of shrinking the map:

- S7a encoder records the jitcode pc in resume data (alongside py_pc initially,
  probe-compared).
- S7b guard-liveness: `get_list_of_active_boxes`/`generate_guard` read liveness at
  the current jitcode position during tracing (removes the carry-forward dependency,
  trace_opcode.rs:1972/1999).
- S7c blackhole/bridge resume entry keys by jitcode pc (jitcode_dispatch.rs:8383/8657
  inverse lookups retired).
- S7d delete `pc_map`, `derive_pc_live_indices_from_sparse` (codewriter.rs:773/11198),
  and the dense liveness side-tables.

Est. 2–4 sessions. Depends on Phase 1 (pcdep keying) and benefits from Phase 6
(decoder already positional).

## Phase 8 — siblings and cleanup

- #359 residual exc/resume siblings (M2–M6, 9 mechanisms) — re-triage after Phase 4;
  several are expected to collapse with the recipe/decline retirement.
- #339, #329 — close via Phase 4c; verify with their archived repros.
- #369 pop_top arm-id test fixes (mechanical).
- The 2 flaky short-circuit SIGSEGVs and the flatten.rs make_return arity-8 panic
  (pre-existing finds from the #370 flip WF) — file/fix opportunistically.
- PR consolidation (push is user-gated; never push).

## Dependency graph

```
P1 (mirror capture totality at kept-stack guards; absorbs old P2)
 ├─→ P3 (delete 2 bounded flat readers)
 ├─→ P4 (encoder/bridge/recipe → pcdep; #124 core; unblocks #329/#339)
 │       └─→ P5 (delete stack_slot_color_map)
 └─→ P6 (Path 4 #238: decode overlay → regalloc de-pinning → local map)
         └─→ P7 (#73 pc_map retirement — the ultimate parity endpoint)
P8 rides on P4/P6.
```

Note P4's own merge-PC pcdep gap (decode-side) may resurface when migrating the
recipe/encoder; handle it THERE with walker-level Variable-ization if needed — not as
a prerequisite for P1/P3.

Total rough estimate: 12–20 sessions. The single highest-leverage next action is
Phase 1 S1a (the bucket census) → S1b [EDGE-COPY] (mirror-native #420 recovery).
