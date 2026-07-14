# pyre Rework Program

**Status**: proposed. Companion to `design.md` (the charter). Where the
charter states what pyre must be, this document states where today's code
violates it, with evidence, and defines the corrective program. It was
produced by auditing the tree (branch `pc-map`, 2026-07-05) against the
charter's axioms A1–A7 and norms N1–N7. Evidence citations are of two kinds:
code locations verified in this audit, and the issue/epic record (memory
files, PRs) for damage history.

The verdict up front: **the skeleton is right, the JIT spine is where the
violations live.** The layer map of charter §1 is real in the tree; none of
the anti-roadmap (§3.5) items have been rebuilt. The structural defects are
concentrated in four places, all inside trace/resume/translate/GC-roots —
exactly the Phase A territory the charter says outranks everything.

---

## 1. Findings

### F1 — The snapshot/resume machinery invented its own coordinate system

**What exists.** `SnapshotFrame` stores the *Python bytecode* PC where
RPython's `resume.py` stores the JitCode offset, and recovers the JitCode
position through a lossy `pc_map` built at trace time
(`majit-metainterp/src/pyjitpl.rs:506`, readers at
`pyre-jit-trace/src/state.rs:1009–1248`, `resumedata.rs:84`). Bolted on top
is a second, correct coordinate — a per-frame `jitcode_pc: i32` word
(`resume.rs:279`) with a `NO_JITCODE_PC` sentinel — which branch-guard
resume now prefers (#73/#366, gates already removed). So one snapshot
carries **two coordinate systems**, one of them lossy, plus a translation
layer that PyPy never needed.

**Violates.** A6/N2 (parity: no invented representations), and via its
consequences A1 (traces that resume at the wrong position are semantics the
interpreter never had).

**Evidenced damage.** This is the root of the worst shipped-bug class:
the FOR_ITER "not an iterator" crash live on origin/main (guard-failure
resume writes a green pycode into a red iterator slot), the loop-carried
`or` deopt underflow, and the whole task50/#215 slot-vs-color epic. Each
was a symptom of resume reconstructing state through the invented
coordinates instead of resume.py's.

**Correct shape.** `rpython/jit/metainterp/resume.py`: snapshot frames
keyed by JitCode position natively; numbering via `rd_numb`/numb_state;
`rebuild_from_resumedata` as the single reconstruction path. pyre already
has the analog entry points (`eval.rs:6429–6743`, `dispatch_perfn_frame`,
`setup_reconstructed_callee_frame`) — the representation underneath must
converge.

**Tracking.** Fully issued: gh#366 (tracer interprets JitCode directly —
the enabling engine), gh#368 (delete `pc_map` + the legacy Python-pc resume map,
Artifact 1), gh#369 (retire the carried `jitcode_pc` side-channel,
Artifact 3), gh#367 (bank-aware pcdep color-slot map, Artifact 2 /
slot-vs-color); groundwork gh#73 + PR#365 (closed, M3 milestone). Related:
gh#343 (multi-frame virtual-PyFrame rematerialization on deopt),
gh#371 (compile-time walker coalescing residual).

### F2 — Trace time has a hand-written interpreter twin (three executors)

**What exists.** PyPy has two executors: metainterp (trace time) and
blackhole (deopt time). pyre has three: inside trace time, a walker leg
(`full_body_walk_trace` / `run_perfn_walk`,
`pyre-jit-trace/src/jitcode_dispatch.rs:466,2265,2464`) coexists with a
trait leg — `OpcodeHandler` impls on `MIFrame` that the code itself
describes as "the trace-time twin of PyFrame's impls in pyre-interpreter"
(`pyre-jit-trace/src/lib.rs:87`). Virtualizable handling stays on the trait
leg because the walker cannot observe a force without executing the callee.

**Violates.** A1 directly. A hand-written trace-time twin *is* a JIT with
semantics of its own; every divergence between the twin and PyFrame is a
miscompile waiting for an input. Two trace-time legs over the same bytecode
double the divergence surface.

**Evidenced damage.** walker BYPASS misfire under suspend → stack
underflow (task#29), wasm re-entry corruption by the walk loop (timeout
BUG#2), aheui never-close. Each was the two legs disagreeing.

**Correct shape.** One trace-time executor. The W4 single-executor epic
(PR#311, walker-as-tracer) already points here; it must finish, and the
trait twin must be deleted (A7), with the vable-force blocker solved the
way PyPy's metainterp does it rather than by keeping a second leg.

**Tracking.** gh#344 (observer/replay two-executor → single authoritative
walker) is the epic; its original scope was the generic majit engine only,
and the pyre-side half — deleting the `OpcodeHandler` trace-time twin and
the `is_full_body_walk` bifurcation — is now recorded there as a scope
supplement (2026-07-05 comment). gh#342 and gh#115 track the walker
coverage gaps that force the trait leg to stay alive.

### F3 — GC root registration is a post-hoc walker registry

**What exists.** `MAX_EXTRA_ROOT_WALKERS = 16`
(`majit-gc/src/shadow_stack.rs:681`), a fixed array that panics on
overflow, currently holding **14 walkers**: 13 registered from
`pyre-jit/src/eval.rs:2106–2247` (rd_consts, partial/active trace, compile
snapshot, jitcode constants, FBW journals ×2, interpreter side table,
signal handlers, weakref boxes, sre patterns, jit callee frames, pyre
objects) plus the gc-table walker (`majit-gc/src/gcreftracer.rs:175`).

**Violates.** A2 (memory policy woven, not accreted). Nearly every walker
was added *after* a use-after-free (signal handlers #30, weakref boxes #31,
sre patterns #29, weakref registration #188…). Each is a confession that
some object lives outside GC discipline as an untracked immortal; the
registry is reactive whack-a-mole with a hard cap.

**Correct shape.** incminimark's model: shadow stack for stack roots,
the prebuilt-object protocol for immortals, GC-traced frames for the JIT
(the resolution D05.x itself predicted — "let the JIT find roots since it
knows frame layout"). gh#355 (PyFrame→W_Root, landed through S2) is the
template: move each walker's object population under normal GC tracing and
delete the walker.

**Tracking.** gh#355. Originally it covered frames only; the full registry
retirement (taxonomy of the 14 walkers, prebuilt-protocol absorption of the
immortal populations, shrinking `MAX_EXTRA_ROOT_WALKERS` as the progress
metric) is now recorded there as a scope supplement (2026-07-05 comment).

### F4 — majit-translate coverage is sustained by per-case seams

**What exists.** Rust idioms still lack systematic lowering: the #346
generic `E::Value` monomorphization gap diverges ~124 opcode graphs
(front-end abstract-shell on multi-impl traits); the #131 Result/Option
work concluded its clean front-synth seam is **exhausted** (remaining:
UnionError lattice, Vec/IndexMap, boxing, iterators). Constructs that don't
lower degrade not into tracked residual calls but into `abort_permanent`
**compilation cliffs** — the gh#373 class, where a hot loop silently never
compiles (demonstrated 32×/20× cliffs; latent no-token variants).

**Violates.** A1 ("Rust can't be meta-traced is never a valid excuse") and
charter §3.1's norm that every fallback is a census-tracked gap, never a
silent hole.

**Correct shape.** The rtyper/exceptiontransform parity ports already
chosen: Epic A approach-1 (unique-KIND monomorphization) for #346, Option A
shape-agnostic exceptiontransform for #131, with the census workflow as the
completeness instrument and `abort_permanent` reduced to a small, listed,
justified residue.

**Tracking.** Well issued: gh#346 (two-phase coverage roadmap, the epic;
successor of the closed gh#131) with per-gap instance issues gh#336
(rrange), gh#337 (rlist), gh#182 (rbuiltin typers), gh#339 (boxing
NewWithVtable), gh#181 (box_value Void), gh#176 (phi threading), gh#180
(gctransform), gh#139 (EffectInfo). The cliff symptom is gh#373.

### F5 — Deficiencies (right design, unfinished) — the debt list

- **Compilation cliffs**: unported opcode classes (CallIntrinsic2, GetLen,
  LoadSpecial — the CALL_INTRINSIC_1 fix `f0f68547ba` is the template;
  gh#373), nested-loop/cross-loop no-token walls (gh#152, gh#177;
  cross-loop-cut S0–S3 approved), recursion/call-frame wall (gh#126,
  gh#343; the closed gh#215 was the umbrella).
- **Gate debt**: **119 distinct `PYRE_*` env gates** in the tree (28 in the
  FBW family alone). Charter §3.6: a gate is a staging area, not a home.
  No triage table exists. *No tracking issue.*
- **Documentation rot (N7)**: `majit/README.md` documented the deleted
  majit-analyze era (crates majit-opt/meta/codegen/runtime/analyze vs the
  actual majit-translate/metainterp/backend-* tree). **Resolved 2026-07-05:
  rewritten to the actual crate tree and pipeline.** (The suspected
  duplicate `readme.md` was a case-insensitive-filesystem artifact; only
  `README.md` is tracked.)
- **Phase C debt with a deadline flavor**: no C-extension strategy decision
  document. The EU final report's admitted decade-costing error is exactly
  this deferral; the charter (§5 Phase C) requires a decision document —
  writing it does not require Phase A to be finished. Tracked as gh#376.

### Explicit non-findings (audited and ruled parity or justified)

- `MIFrame` as a type distinct from `PyFrame` is **parity** (PyPy's
  pyjitpl has MIFrame); the defect is the hand-written *OpcodeHandler twin*
  on it (F2), not the type split.
- Snapshots staying `Box` while frames became W_Root (#355 policy) —
  deliberate, documented, keep.
- TLS singletons (BACK_EDGE_BH_BUILDER etc.) — audited against PyPy's
  GIL-justified singletons and documented per charter §3.3; keep.
- Thin backends behind one trait (dynasm primary, Cranelift, wasm) — charter
  §3.4 answer; not rework territory even where compile latency hurts
  (recourse ladder applies).

---

## 2. Workstreams

### WS1 — Trace/resume convergence (F1 + F2) — *the* priority

Goal: one trace-time executor, one resume coordinate system, both at
resume.py/pyjitpl.py parity. F1 and F2 are one workstream because the
resume representation is written by the tracer: retiring pc_map requires
every snapshot writer to know its JitCode position, which is what the
single-executor walker provides.

Increments (each lands green on N4 gates, each with its kill switch):

1. **Finish gh#366 direct-pc coverage**: extend `jitcode_pc` capture to the
   remaining guard classes (GuardNoException/GuardNotForced), flip default.
   Exit: every snapshot frame carries a valid direct coordinate.
2. **Invert the representation** (gh#368 + gh#369): make the JitCode offset
   the primary `SnapshotFrame` field (resume.py shape); delete the
   Python-PC field, the snapshot `pc_map` (pyjitpl.rs:506), and the
   resume translation layer (gh#368), then the carried
   `jitcode_pc` side-channel it obsoletes (gh#369). Python-level PC, where
   genuinely needed (frame f_lasti, tracebacks), is derived the way PyPy
   derives it, not stored as the resume key.
3. **Numbering parity**: converge serialization on resume.py's
   rd_numb/numb_state tagged numbering, including non-Ref banks (gh#367);
   complete rebuild_from_resumedata parity for multi-frame reconstruction
   (gh#343; the dispatch_perfn_frame / setup_reconstructed_callee_frame
   scaffolding exists; the slot-vs-color seeding panic is the open edge).
   *The full-numbering-parity umbrella itself has no issue* — file one if
   residue remains after gh#367/gh#368/gh#369 close.
4. **Single executor (W4)**: make walker-as-tracer the only trace-time
   leg; solve the vable-force observation blocker via the metainterp
   mechanism; delete the `OpcodeHandler` trace-time twin and the
   `is_full_body_walk` bifurcation (A7). gh#344 owns both halves (the
   pyre-side twin deletion was added to its scope 2026-07-05);
   gh#342/gh#115 track the walker coverage gaps.

Regression corpus (all must be tests before the increments that fix them):
the pr354 FOR_ITER-in-called-function crash repro, the loop-carried `or`
deopt underflow repro, rc_d32.py double-append, aheui logo --jit, the wasm
timeout re-entry cases.

Exit criteria: `pc_map`, the resume translation layer, `OpcodeHandler` twin, and
`is_full_body_walk` no longer exist in the tree; full benchmark suite (all
8) no regressions; crash corpus green; slot-vs-color epic closeable.

### WS2 — majit-translate systematization (F4)

Goal: no silent cliffs; Rust-idiom lowering is systematic, census-driven.

1. **#346 Epic A** (unique-KIND generic monomorphization) to close the
   ~124-graph divergence — this is the current largest single source of
   un-lowered bodies.
2. **#131 Option A** shape-agnostic exceptiontransform through the
   remaining inventory (UnionError lattice, Vec/IndexMap, boxing,
   iterators), in census order.
3. **Cliff conversion**: every remaining `abort_permanent` source either
   ports (the CALL_INTRINSIC_1 → HLOp template: CallIntrinsic2, GetLen,
   LoadSpecial) or becomes a *tracked* residual call in the census with an
   owner issue. The latent gh#373 no-token variants get repros first.

Exit criteria: census reports zero unlisted abort_permanent sources; the
known 32×/20× cliff benchmarks compile; gh#346 (and its instance issues)
closed or reduced to listed residue. Parallelizable with WS1 (different
crates, different people-time).

### WS3 — GC roots rework (F3)

Goal: retire the extra-root-walker registry by absorbing its populations
into normal GC discipline. Tracked in gh#355 (scope supplemented
2026-07-05 to cover the full registry retirement).

1. **Taxonomy pass**: classify the 14 walkers into (a) trace/JIT state
   that belongs in GC-traced structures (rd_consts, partial/active trace,
   compile snapshot, jitcode constants, callee frames — the #355/W_Root
   track), (b) interpreter-global populations that belong on the
   prebuilt-object protocol (signal handlers, sre patterns, side tables),
   (c) genuine GC-internal tables (gcreftracer) that stay.
2. **Absorb class (a)** along the #355 S2c/S3/S4 track (arena task#7,
   typedef, stub retirement) — this work is already sequenced.
3. **Absorb class (b)** via the prebuilt/immortal protocol with traced
   children (the #29 immortal-children fix generalized), deleting each
   walker as its population moves.
4. Shrink `MAX_EXTRA_ROOT_WALKERS` as walkers retire — the shrinking
   constant is the progress metric; the panic-on-overflow branch should
   become unreachable and then deleted.

Verification: the GC probe suite, nursery-stress oracle
(small-nursery runs, the PYPY_GC_NURSERY=131072 technique), and the
regrtest harness under moving collection (the oldgen-nonmoving concession
should become deletable — that is the real exit test).

### WS4 — Hygiene batch (F5, continuous)

1. **Gate triage** (done 2026-07-05): `gate-triage.md` classifies all
   ~119 `PYRE_*` matches. Findings: ~20 are not gates (Rust
   identifiers) or dead (no read site); ~99 real env vars, ~33 default-ON
   experiments. The **wasm trio** (`PYRE_WASM_CA`, `_ENABLE_BRIDGES`,
   `_INLINE_ALLOC`) was retired (hardwired ON, machinery deleted, verified
   compile-clean native + wasm32). The other ~30 default-ON gates are
   load-bearing kill switches for open reworks (FBW/executor #344/#366,
   rtyper #346, GC #355, for-iter #57) — each retires when its epic closes
   (A7). See `gate-triage.md` §4 for the per-gate retire trigger.
2. **README rewrite**: `majit/README.md` to the actual crate tree.
   **Done 2026-07-05.**
3. **Phase C decision document** (gh#376): C-extension strategy
   (HPy/cpyext-class options, rctypes failure and cpyext cost curve as
   priors) — a document, not an implementation; unblocks nothing but
   forgets nothing.

---

## 3. Sequencing and interaction

Priority under contention (charter §5 order): **WS1 > WS2 > WS3 > WS4**,
with the qualifications:

- WS1 increments 1–3 are the critical path — they root out the shipped
  miscompile class. Increment 4 (W4) is the largest single piece; its
  prerequisite work (per-opcode entry_py_pc advance, concrete seeding) is
  already landed on the for-iter/rewrite-tracer lines.
- WS2 is parallel-safe with WS1 (majit-translate vs
  majit-metainterp/pyre-jit-trace) and is the precondition for Phase A's
  cliff-free exit criterion.
- WS3 rides the already-sequenced #355 track; its class-(b) work is
  independent and can interleave.
- WS4 items 1–2 are cheap and immediate; the gate triage should happen
  *early* because WS1/WS2 will otherwise keep adding gates to an untriaged
  pile (every WS increment's kill switch enters the table at birth, with
  its flip-or-delete date).

Each workstream closes by the charter's own instruments: N4 gates for
every landing, N5 evidence for every default flip, N7 written rationale
(epic memory file or issue) for every mechanism deleted or replaced.

---

## 4. What falsifies this program

Per charter §6, the program is amendable by evidence. Specifically:

- If WS1 increment 2 shows the Python-PC field is load-bearing for
  something PyPy handles differently at a structural level (not a bug),
  the finding F1 remedy narrows to "keep derived, not stored".
- If the W4 vable-force blocker proves to require metainterp behavior that
  the walker architecture cannot express, F2's remedy escalates from
  "finish W4" to "re-evaluate the walker against a straight pyjitpl port"
  — a bigger rework, to be proposed separately with the evidence.
- If WS3's class-(b) absorption measurably regresses minor-collection
  pause (prebuilt scanning cost), the registry survives *for that class
  only*, documented as the deliberate adaptation it currently isn't.
