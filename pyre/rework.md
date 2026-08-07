# pyre Rework Program

**Status**: living record, companion to `design.md` (the charter). Where the
charter states what pyre must be, this states where today's code violates it and
what is left to do about it. **Findings are deleted as they close** — the history
of what was once wrong belongs in git, not here. The one exception is *Settled*
below: a closed finding leaves a one-line verdict there only where re-deriving it
is the live risk. Keep this document small enough that it is worth re-reading.

Original audit: branch `pc-map`, 2026-07-05, against the charter's axioms A1–A7
and norms N1–N7. Re-measured 2026-08-07 on `ec-wiring`.

The verdict has not changed shape: **the skeleton is right, the JIT spine is
where the violations live.** The layer map of charter §1 is real in the tree and
none of the anti-roadmap (§3.5) items have been rebuilt. What changed is the size
of the remainder — two of the original five findings are closed and deleted, and
thirteen of the fifteen tracked issues are closed.

---

## Open findings

### F4 — one `_other` catch-all is the whole unlisted-cliff surface

**Counted 2026-08-07.** Earlier revisions of this finding reported "~214 matches,
unchanged in scale" and left it unmeasurable behind a census that was never
built. Counting *emission sites* rather than mentions changes the picture:
the 219 textual matches across 20 files are almost all comments. The real surface
is **23 `emit_abort_permanent!` sites, all in `pyre-jit/src/jit/codewriter.rs`**
— 22 named-opcode arms plus one catch-all — and every named arm already carries
its reason inline.

| class | n | opcodes |
|---|---|---|
| **genuine trace boundaries** — a trace records one continuous execution, so no residual can express the resume | 9 | `YieldValue`, `Send`, `EndSend`, `ReturnGenerator`, `GetYieldFromIter`, `GetAiter`, `GetAnext`, `EndAsyncFor`, `CleanupThrow` |
| **narrow conditional shapes** inside an otherwise-lowered opcode | 5 | `Call` / `CallKw` (nargs past the backend dispatch ceiling), `LoadFastCheck`, `LoadLocals` (non-portal `is_locals`), `DeleteDeref` (compiler-normalized class-scope case) |
| **unported opcodes** — the real coverage gap | 8 | `CheckEgMatch`, `BuildInterpolation`, `BuildTemplate`, `CallIntrinsic1`, `CallIntrinsic2`, `LoadSpecial`, `LoadFromDictOrDeref`, `SetupAnnotations` |
| **catch-all** `_other => emit_abort_permanent!(py_pc)` | 1 | whatever the match does not name |

**Violates.** A1 ("Rust can't be meta-traced is never a valid excuse") and
charter §3.1's norm that every fallback is a census-tracked gap, never a silent
hole.

**Where the violation actually lives.** Not in the 22 — those are listed by
construction, each beside its reason. It lives in the **`_other` arm**: an opcode
nobody has looked at declines the whole loop with no record of which opcode it
was. That single arm is the entire unlisted surface, which is why the finding
could never be closed by counting matches.

**What is left.**

1. **Make `_other` name its opcode.** A cliff that says which instruction caused
   it stops being a silent hole, and the gap list below becomes self-maintaining
   instead of needing a census run to rediscover.
2. **Port the 8.** `CALL_INTRINSIC_1 → HLOp` is the template; `CallIntrinsic1`
   still aborts on the intrinsic kinds that arm does not cover.
3. **Record the 9 as boundaries, not gaps.** They are correct, and counting them
   is most of what made the raw number look like a wall.

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

### F5 — three documented gates have no reader left, and the brake is Rust-only

The 66 undocumented gates and the missing brake are both closed: `gate-triage.md`
§6 lists all 66, and `pyre/pyrex/tests/gate_triage_complete.rs` fails the build
when a `PYRE_*` read in a workspace member has no entry in a **live** section of
that file. **Measured 2026-08-07**, re-measured once §6 landed:

| count | value | what it is |
|---|---|---|
| distinct names read from `*.rs` | **105** | the population the brake sees |
| (file, name) read pairs | 128 | read *sites*, not gates |
| named in `gate-triage.md`'s live sections | 113 | tokens; one is the `PYRE_FBW_*` fragment in §1d's heading |
| named in its retirement sections | 47 | already swept |
| **live-named with no reader anywhere** | **3** | the debt |

```sh
git ls-files '*.rs' | xargs rg --no-filename -o \
  '(env::var[_a-z]*|host_os::var|getenv)\(b?"(PYRE_[A-Z0-9_]+)"' -r '$2' | sort -u
```

Say which of these a number is. Earlier revisions reported 119 and then 126
"distinct names"; both were the (file, name) pair count, because the command as
written then kept rg's filename prefix and `sort -u` counted sites.

**Violates.** Charter §3.6: a gate is a staging area, not a home. Three names are
staged in a file nobody swept.

**What is left.**

1. Retire `PYRE_FBW_REC_UNROLL`, `PYRE_FBW_VABLE_SCALAR_CA` and `PYRE_P2_DRAIN`
   — named in §5/§1d, read from nothing in the tree.
2. Decide whether the brake scans beyond Rust. Four live gates
   (`PYRE_CHECK_PYPY3`, `PYRE_CHECK_PYTHON3`, `PYRE_SHARED_BUILD`,
   `PYRE_SYNTH_PYPY`) are read only by `check.py`, `check_synthetic.py`, the CI
   workflows and `scripts/llbc_extract.py`. A `*.rs` census reads them as retire
   targets and they are not; conversely a Python- or YAML-only gate added
   tomorrow enters undocumented, which is the hole §6 was written to close.

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

**F4 > F3 > F5.** The charter's own §5 order, restored — it was inverted in an
earlier revision because the root-walker registry had a hard failure one
registration away, and that is gone.

- **F4 first**, and it is now small. Naming the opcode in the `_other` arm is a
  one-site change that converts the only silent hole into a reported one; the
  eight unported opcodes are then a list somebody can work through, and Phase A's
  cliff-free exit criterion becomes checkable without any new instrument.
- **F3 next**: the deepest structural work, unblocked by one answerable question
  (the mid-major rescan above) rather than by a taxonomy.
- **F5 out of order whenever convenient** — it is cheap, it is the only item that
  gets *worse* while ignored, and its brake is what keeps it closed.

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

- The F4 census is built, and the unlisted set is not empty: it is the one
  `_other` catch-all. If naming that arm's opcode shows it never fires on the
  corpus, F4 closes on the spot and only the tracked residue remains.
- If F3's class-(b) absorption measurably regresses minor-collection pause
  (prebuilt scanning cost), the registry survives *for that class only*,
  documented as the deliberate adaptation it currently is not.
