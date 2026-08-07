# pyre Rework Program

**Status**: living record, companion to `design.md` (the charter). Where the
charter states what pyre must be, this states where today's code violates it and
what is left to do about it. **Findings are deleted as they close** — the history
of what was once wrong belongs in git, not here. The one exception is *Settled*
below: a closed finding leaves a one-line verdict there only where re-deriving it
is the live risk. Keep this document small enough that it is worth re-reading.

Original audit: branch `pc-map`, 2026-07-05, against the charter's axioms A1–A7
and norms N1–N7. Re-measured 2026-08-08 on `ec-wiring`.

The verdict has not changed shape: **the skeleton is right, the JIT spine is
where the violations live.** The layer map of charter §1 is real in the tree and
none of the anti-roadmap (§3.5) items have been rebuilt. What changed is the size
of the remainder — three of the original five findings are closed and deleted,
and thirteen of the fifteen tracked issues are closed.

---

## Open findings

### F4 — eight unported opcodes; the unlisted surface is gone

**Counted 2026-08-07, re-measured 2026-08-08.** Earlier revisions reported "~214
matches, unchanged in scale" and left the finding unmeasurable behind a census
that was never built. Counting *emission sites* rather than mentions changed the
picture: the textual matches across 20 files are almost all comments, and the
real surface is the `emit_abort_permanent!` sites in
`pyre-jit/src/jit/codewriter.rs` — now **25**, every one of them named.

| class | n | opcodes |
|---|---|---|
| **genuine trace boundaries** — a trace records one continuous execution, so no residual can express the resume | 9 | `YieldValue`, `Send`, `EndSend`, `ReturnGenerator`, `GetYieldFromIter`, `GetAiter`, `GetAnext`, `EndAsyncFor`, `CleanupThrow` |
| **narrow conditional shapes** inside an otherwise-lowered opcode | 5 | `Call` / `CallKw` (nargs past the backend dispatch ceiling), `LoadFastCheck`, `LoadLocals` (non-portal `is_locals`), `DeleteDeref` (compiler-normalized class-scope case) |
| **unported opcodes** — the real coverage gap | 8 | `CheckEgMatch`, `BuildInterpolation`, `BuildTemplate`, `CallIntrinsic1`, `CallIntrinsic2`, `LoadSpecial`, `LoadFromDictOrDeref`, `SetupAnnotations` |
| **not emittable by this compiler** — three classified arms where the catch-all was | 108 | 82 adaptive specializations, 21 `Instrumented*`, 5 interpreter/JIT-internal |

**Violates.** A1 ("Rust can't be meta-traced is never a valid excuse") and
charter §3.1's norm that every fallback is a census-tracked gap, never a silent
hole.

**What the catch-all turned out to be.** Not "an opcode nobody has looked at" —
`_other` covered exactly the 108 `Instruction` variants the dispatch never named,
and all 108 are opcodes this compiler cannot emit: the adaptive specializations a
quickening interpreter writes in place (pyre's eval loop does not quicken —
nothing calls `replace_op` outside a corruption test), the `sys.monitoring`
substitutions, and the tier-2 executor's internal forms. So the unlisted cliff
was never a live cliff. It was the *silence* that mattered: had one appeared, the
loop declined with no record of which opcode did it.

Deleting `_other` and classifying all 108 makes the match exhaustive, so a
variant added upstream now fails to compile here instead of vanishing into a
catch-all — `majit-translate`'s `flowspace/flowcontext.rs` already classifies the
same three groups, and this brings the walker in line with it. `cargo check -p
pyre-jit` passes with no catch-all, and dropping a single pattern from the list
fails as `E0004: non-exhaustive patterns: Instruction::ToBoolBool not covered`,
which is what makes the coverage claim load-bearing rather than decorative.

**What is left.** **Port the 8.** `CALL_INTRINSIC_1 → HLOp` is the template;
`CallIntrinsic1` still aborts on the intrinsic kinds that arm does not cover. The
9 boundaries are correct and counting them is most of what made the raw number
look like a wall.

**Tracking.** gh#346 (two-phase coverage roadmap) and gh#373 (the cliff symptom)
are closed; coverage work continues against the #346 line.

### F3 — GC roots are walked by an embedder registry, not absorbed into GC discipline

**What exists.** `majit-gc`'s `EXTRA_ROOT_WALKERS` is a fixed array the embedder
plugs callbacks into, walked from `do_collect_nursery`'s Phase 1e. **Four sources
are registered** (of a cap of 8): the interpreter's process-global off-GC slots
(`walk_interpreter_global_roots`), the exceptions parked outside GC discipline
(`walk_parked_exception_roots`), the immortal process-global stores
(`walk_immortal_store_roots`), and the per-loop gc_table walker in
`majit-gc/src/gcreftracer.rs`.

**Violates.** A2 (memory policy woven, not accreted). Almost every population
inside those four sources was added *after* a use-after-free — signal handlers,
weakref boxes, sre patterns, the immortal exception singletons' children. Each is
a confession that some object lives outside GC discipline as an untracked
immortal, and grouping them by storage kind did not change that.

**Correct shape.** incminimark's model: shadow stack for stack roots, the
prebuilt-object protocol for immortals, GC-traced frames for the JIT.
`framework.py root_walker.walk_roots` registers a *fixed* set of root-storage
kinds; it has no per-data-structure callback array at all.

**What is left.**

1. **Class (a) — trace/JIT state** into GC-traced structures. Largely done: the
   per-mutator `MutatorEntry.extra_areas` already carries frame roots, jitcode
   constants, the FBW journals, mapdict and callee frames.
2. **Class (b) — interpreter-global and parked-exception populations** onto the
   prebuilt/immortal protocol with traced children, deleting each carrier as its
   population moves.
3. **Class (c) — `gcreftracer`** is genuinely GC-internal and stays.

The concrete next step is already scouted. `BH_LAST_EXC_VALUE`,
`GUARD_EXC_VALUE` and `TL_JIT_PENDING_EXCEPTION` are **already** walked by the
per-mutator `PyFrameRootArea`: `walk_pyframe_roots_area` forwards all three
cells (alongside the in-flight exception and the pending call/hash errors), and
the collector reaches it through `walk_all_extra_areas` or `walk_my_extra_areas`
— both of which include the collecting thread. So those three entries in
`walk_parked_exception_roots` are duplication on every path *except*
`rescan_major_nonstack_roots_and_drain`, which re-walks `walk_extra_roots` but
deliberately not the per-mutator areas ("upstream repeats only
`collect_nonstack_roots`, not `collect_roots`"). **Whether a TLS exception cell
needs that mid-major rescan is the single question standing between here and
deleting three of the seven carriers** — and it decides the shape of the rest of
class (b), because every carrier there is thread-local.

**Progress metric.** `MAX_EXTRA_ROOT_WALKERS`, which shrinks as sources are
absorbed; the panic-on-overflow branch should become unreachable and then be
deleted. Raising it is the accretion this finding condemns — a source that has
nowhere to go belongs inside an existing kind, not in a new slot.

**Verification.** The GC probe suite, the nursery-stress oracle (small-nursery
runs), and the regrtest harness under moving collection. The real exit test is
that the oldgen-nonmoving concession becomes deletable.

### Smaller open items

- **One unproven resume coordinate.** `build_state_field_snapshot` stamps
  `py_pc: frame.pc` — the JitCode offset — into the field whose readers in
  `pyre-jit/src/eval.rs` treat it as a Python pc (`f_lasti`, traceback
  reconstruction, the pcdep colour lookup). The walker's own
  `build_framestack_snapshot` does the opposite, and correctly: it resolves the
  Python pc from the codewriter's marker table and *declines the trace* rather
  than publish a fallback. The corpus is green, so if the state-field writer's
  value does reach frame reconstruction the damage is latent — a mismatched
  colour lookup returns `None` rather than crashing. **Needs a runtime probe**
  (cross-check the decoded `py_pc` against
  `containing_py_pc_for_jitcode_pc_public(jitcode_index, pc)` over the corpus)
  before it is a finding rather than a suspicion.
- **A misleading survivor of the resume rework.** `pyjitpl.rs` still calls its
  `SnapshotFramePcs` local `pc_map`, the name of the deleted translation table.
  Rename it so the name stops implying a mechanism that no longer exists.
- **Compilation cliffs** outside F4's census: nested-loop / cross-loop no-token
  walls (gh#152, gh#177) and the recursion / call-frame wall (gh#126, open).
- **Phase C decision document** (gh#376, open): a C-extension strategy document,
  not an implementation. Writing it does not require Phase A to finish, and the
  EU final report's admitted decade-costing error is exactly this deferral.

---

## Sequencing

**F4 > F3.** The charter's own §5 order, restored — it was inverted in an earlier
revision because the root-walker registry had a hard failure one registration
away, and that is gone.

- **F4 first**, and it is now small: eight unported opcodes, a list somebody can
  work through. The silent hole is closed, so Phase A's cliff-free exit criterion
  is checkable without any new instrument — an opcode the walker declines names
  itself, and one it has never seen fails the build.
- **F3 next**: the deepest structural work, unblocked by one answerable question
  (the mid-major rescan above) rather than by a taxonomy.

F4 and F3 are parallel-safe, though not for the reason an earlier revision gave:
both touch `pyre-jit`, so "different crates" was wrong. They share no file and no
symbol — F4 is confined to `jit/codewriter.rs`, F3 to the root-walker
registration in `eval.rs` / `call_jit.rs` and `majit-gc/shadow_stack.rs`.

Each item closes by the charter's instruments: N4 gates for every landing, N5
evidence for every default flip, N7 written rationale for every mechanism deleted
or replaced. **And then its section here is deleted.**

---

## Settled — do not re-litigate

**Closed findings.**

- *Resume coordinates invented their own system.* The lossy `pc_map` translation
  and the duplicate Python-PC coordinate are gone: `metadata.pc_map` and
  `resume_jitcode_pc_for` have zero hits, and `SnapshotFrame.pc` is the JitCode
  byte offset at every writer. The `pc_map` name that survives in
  `jit/codewriter.rs` / `jit/flatten.rs` is an unrelated compile-time
  `Vec<usize>` driving exit-recovery construction.
- *Three trace-time executors.* `is_full_body_walk`, `PYRE_FULL_BODY_WALK` and
  the `OpcodeHandler`-on-`MIFrame` twin are deleted — zero hits each. The walker
  is the sole trace-time executor and observes a vable-force through the
  residual-call token protocol, which is the metainterp mechanism, not a second
  leg.
- *Gates staged in a file nobody swept.* The 66 undocumented gates, the missing
  brake, the brake's Rust-only reach and the last reader-less entry are all
  closed: 111 names read (105 from `*.rs`, 6 only from the harness) and the same
  111 documented live, once the two `PYRE_*` wildcard stems the token scan leaves
  behind are set aside.
  `pyre/pyrex/tests/gate_triage_complete.rs` now fails the build in **both**
  directions — a read with no live entry, and a live entry with no reader. Say
  which of these a number is: earlier revisions reported 119 and then 126
  "distinct names" and both were the (file, name) pair count, because the census
  command kept rg's filename prefix and `sort -u` counted sites.
  `gate-triage.md` §6 carries the command and the counting rules; the two that
  cost the most to rediscover are that a retirement row documents nothing
  wherever it sits, and that a name written only in the file's `_PYRE`,
  `_PYTHON` run-on shorthand reads as undocumented to any whole-token census.

**Deliberate adaptations, decided — keep.**

- `SnapshotFrame.py_pc` is **carried, not derived**. The derivation exists
  (`py_coord::containing_py_pc_for_jitcode_pc`) and is the dominant mechanism
  everywhere else, but the resume decoder that reads `py_pc` back holds no
  jitcode metadata and so has no inverse available at that point.
- `MIFrame` as a type distinct from `PyFrame` is parity — pyjitpl has MIFrame.
  The defect was the hand-written OpcodeHandler twin on it, now deleted.
- Snapshots stay `Box` while frames became `W_Root`.
- TLS singletons (`BACK_EDGE_BH_BUILDER` and friends), audited against upstream's
  GIL-justified singletons.
- Thin backends behind one trait (dynasm primary, Cranelift, wasm) — charter
  §3.4's answer, not rework territory, even where compile latency hurts.
- `majit/README.md` was rewritten to the actual crate tree.

---

## What falsifies this

- The F4 census is built and the unlisted set is empty: the catch-all held only
  opcodes this compiler cannot emit, and the match is exhaustive without it. What
  remains of F4 is the eight unported opcodes, so if a corpus run shows one of
  those eight never fires either, F4 closes on the spot.
- If F3's class-(b) absorption measurably regresses minor-collection pause
  (prebuilt scanning cost), the registry survives *for that class only*,
  documented as the deliberate adaptation it currently is not.
