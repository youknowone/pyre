# pyre Design Charter

**Status**: normative. This document states the design axioms, the layer
architecture, the adaptation decisions, and the macro-strategy that govern
pyre and majit development. It is grounded in three sources, in order of
authority:

1. **The current PyPy/RPython source tree** (`pypy/`, `rpython/` in this
   repository) — the living, corrected result of twenty years of evolution.
   For any mechanism, this is ground truth for *what* to build.
2. **The PyPy EU reports (2004–2007)** — digested in `../eu_report.md`,
   judged in `../eu_report_assessment.md`. The reports are the canonical
   statement of *why* the architecture is shaped this way: the founders'
   reasoning, their measurements, and — read against later history — a
   catalog of dead ends that must not be re-explored. Code tells us what
   PyPy is; the reports tell us what PyPy meant.
3. **pyre's own measured results** — memory files, benchmark history,
   check.py.

When these conflict: current PyPy code beats the EU reports on mechanism;
the EU reports beat folklore on rationale; pyre's own measurements beat both
on what works *in Rust*.

---

## 1. Mission and layer correspondence

pyre's goal, stated once: **put an RPython-equivalent layer (majit) on top of
Rust, and a PyPy-equivalent (pyre) on top of that.**

| PyPy world | pyre world | Notes |
|---|---|---|
| RPython the language | **Rust** | The host language is no longer a Python subset; it is a real language with a real type system. See §3.1. |
| RPython translator (flowspace → annotator → rtyper) | **majit-translate** (`front/ast` → `flowspace/` → `annotator/` → `rtyper/`) over **Charon LLBC** artifacts | Same pipeline, same module names, run at `cargo build` time over extracted `.ullbc` instead of live bytecode. |
| `jtransform`/codewriter → JitCode | **majit-translate `codewriter/`** → JitCode | Same role, at `cargo build` time. pyre additionally runs a *second*, hand-written codewriter over user `CodeObject`s at runtime; see §3.7. |
| `warmspot` — translation-time portal generator | **split**: build-time derivation in majit-translate, hand-written warm entry in pyre-jit | Not a port. `apply_jit` is unwired and `warmspot.rs` is a `pub use` namespace; see §3.7. |
| metainterp, optimizer, resume, blackhole | **majit-metainterp / majit-trace** | Line-by-line port of the *tracing* JIT (pyjitpl5 lineage), not the 2007 PE JIT. The history, optimizer, resume and blackhole halves are what production runs; the `pyjitpl` MIFrame *tracer* is not — `mod pyjitpl` is private, `cpu.rs` calls that tracer retired, and pyre records through the hand-written full-body walker in `pyre-jit-trace/src/jitcode_dispatch/`. That walker is A1 debt of the same class §3.7 tracks. |
| x86/ARM/… hand-written backends (~300k LOC) | **majit-backend-dynasm / -cranelift / -wasm** | Three thin backends behind one trait, current primary dynasm; see §3.4. |
| incminimark GC | **majit-gc** (nursery + oldgen + incremental + card marking) | Port of the winner, not of Boehm/refcount/mark-sweep. |
| sandbox transform | **rsandbox** | Compile-time sandbox aspect. |
| `pypy/interpreter/` + `pypy/objspace/std/` | **pyre-interpreter + pyre-object** | Structural port, same names, same relative locations. |
| CPython 2.7/3.10 compat | **CPython 3.14** compat | Bit-exact; RustPython compiler front-end supplies bytecode. |
| GIL | **no GIL** | See §3.3. |

majit is a general framework: pyre is its primary consumer but majit must
never depend on pyre. Secondary consumers (aheui-mjit, toy interpreters,
wasmi experiments) exist deliberately — they are the generality proof, the
role PyPy's Prolog/JavaScript interpreters played for RPython.

---

## 2. Axioms

These are the EU-era strategic claims that survived twenty years of
evidence (assessment §7). They are not up for casual renegotiation.

**A1 — Single executable specification.** The interpreter source *is* the
semantics of pyre. The JIT is generated from it; it is never written to have
semantics of its own. Any behavioral divergence between compiled traces and
the interpreter is a **generation defect to fix in majit**, never an accepted
limitation, and never justified by "the interpreter is Rust". (This is the
meta-tracing principle; AGENTS.md states the enforcement details, including
frame identity.)

**A2 — Low-level policy is woven, not written.** Memory management, sandbox,
concurrency machinery, code generation, representation tricks live in the
translation/build layer or in majit, not in interpreter source. The EU
reports' strongest empirical finding stands: encoding a low-level decision
throughout an interpreter's source (CPython's refcounting) makes it
effectively unchangeable. pyre-interpreter must stay readable as "a
straightforward Rust program that executes Python bytecode".

**A3 — Runtime information is the optimization engine.** Static analysis
(annotator/rtyper prepass) exists to *generate the machinery*; the machinery
optimizes with runtime facts — tracing, promotion/guards, virtuals,
virtualizables, quasi-immutability. PyPy's own static-PE JIT losing to
tracing is the controlling precedent. Corollary: when choosing between a
smarter compile-time analysis and a runtime guard, default to the guard.

**A4 — Fall back, never restrict.** Full CPython 3.14 semantics by
construction: traces deopt to the interpreter (blackhole/resume) rather than
the language being restricted to what compiles. `sys._getframe`-class
introspection must work. Correctness is bit-exact — no float tolerances, no
"close enough" (root-cause the divergent operation instead).

**A5 — Tests over proofs; measurement over theory.** Every feature and every
bug fix carries a test. Default-flips of optimizations require the full
benchmark suite; negative and null results get recorded, not forgotten.
Expect D06.1's usual outcome — "no clear tendency" — and keep every
experimental mechanism behind a kill switch until evidence flips it.

**A6 — Port the winner, at parity.** The porting target is *modern* PyPy
source, line-by-line, with data-structure parity (no invented side tables,
no "simplified" shapes — AGENTS.md rules). The EU reports document many
mechanisms (multimethods, PE JIT, ootype, CPS stackless, refcounting) that
modern PyPy deleted; those are rationale history, not porting targets.

**A7 — Deletion is part of the method.** PyPy's team deleted its own
flagship implementations when evidence turned, usually for reasons they had
already written down as open issues. pyre inherits this: gated experiments
are cheap to delete; long-lived gates that never flip are debt; a mechanism
kept "because we built it" is a bug in the process.

---

## 3. Adaptations — where pyre deliberately diverges from the reports

The EU insights cannot be applied verbatim; the substrate changed. Each
adaptation below names what is kept and what is replaced.

### 3.1 "Analyse live programs" → analyse extracted artifacts

The reports' core front-end move — run full Python as a *preprocessor*, then
analyse the live image with bounded dynamism — solved a problem Rust does not
have: recovering static structure from an unspecified dynamic language.
pyre replaces it with:

- **Rust's type system** does what the annotator's type recovery did, at
  zero project cost, with real diagnostics. The chronic RPython pains the
  reports admit (no specification, whole-program-or-nothing, first-error-only,
  cryptic messages, RPylint as a band-aid) are structurally absent.
- **Charon `.ullbc` extraction** plays the role of the frozen live image:
  a whole-program low-level view of the interpreter for majit-translate.
  The price is the same one PyPy paid for image freezing — staleness: source
  changes are invisible until re-extraction. This is a permanent operational
  discipline (fingerprint skipping, forced re-extract), not a temporary bug.
- **Bootstrap dynamism** (RPython's metaclass tricks, generated gateway
  classes, memo functions) maps to proc macros and `build.rs` codegen.
  The annotator that remains in majit is the one RPython's *JIT* needed:
  binding-time and representation analysis over low-level bodies — greens vs
  reds, what is promotable, what is elidable — not type inference.

What is *kept* from the front-end story: the fall-back principle. RPython let
un-analysable code fall back to interpretation; majit lets un-lowerable
interpreter constructs fall back to residual calls / `dont_look_inside`.
The norm (A1) is that each such fallback is a tracked gap with a census, not
a silent permanent hole — the whole point of the prepass census workflow.

### 3.2 Hints: need-oriented, few, and load-bearing

The reports' hint philosophy transfers intact and is worth restating because
it constrains API design for majit forever: hints are **need-oriented** —
placed at the few points where runtime constancy is valuable (`promote`,
green fields, `elidable`, merge points, virtualizables) — and an
unsatisfiable requirement must be a loud error, not silent
de-specialization. majit expresses these as proc-macro attributes with the
exact RPython names. Adding a new kind of hint to work around a translator
weakness is the wrong direction; fix the translator (A1).

### 3.3 Threading: the aspect argument, with no-GIL as the target

D05.4 treats the GIL as one pluggable concurrency model among several and
keeps all concurrency policy out of interpreter semantics. pyre accepts the
framing, and **free-threaded is the target, not the present mechanism**. The
same separation argument is what makes that target reachable — because PyPy's
interpreter source never encoded GIL assumptions into *semantics*, a no-GIL
port is even expressible — but what ships today is a port of RPython's own GIL
(`thread_gil.c` → `majit-gc/src/rgil.rs`); see "GC under free threading"
below for why. Consequences that are already policy:

- GIL-dependent RPython machinery is ported, not merely named: `rgil` has
  production call sites, and a thread holds the GIL for as long as it runs
  pyre code. The pieces that remain name-only (heapcache reset on GIL
  release, `release_gil` effect info) are the JIT-side ones.
- Where RPython used ambient singletons justified by the GIL, pyre must
  justify each adaptation explicitly (TLS with a documented PyPy audit —
  the BACK_EDGE_BH_BUILDER precedent), never silently.
- The GC and object model must be designed for concurrent mutators *as an
  aspect-layer concern* (majit-gc), not by sprinkling atomics through
  pyre-object.

**GC under free threading (gh#396).** The core is an unchanged incminimark;
what supplies its exclusive-heap-access requirement has changed once.

- *Settled 2026-07:* a **stop-the-world safepoint harness** — the HotSpot/SGen
  shape, explicitly not PEP 703's non-moving route.
- *Superseded 2026-08:* pyre supplies exclusive heap access with **a port of
  RPython's own GIL** (`thread_gil.c` → `majit-gc/src/rgil.rs`). The harness's
  per-`gc_op` gate could not be made cheap on its own terms — its entry needs
  per-thread state, which on Mach-O is a `_tlv_get_addr` call on every
  allocation — and upstream has no such gate at all (`rg threadlocalref_addr
  rpython/memory/gctransform/framework.py rpython/memory/gc/incminimark.py`
  has no hits: the allocation path reads no thread identity). Holding the GIL
  across pyre code instead measured **−15.0%** on
  `synth/int_mul_ovf_bignum_promote`, 12/12 rounds faster, and makes `gc_op` a
  bare borrow of the singleton. The safepoint harness is still in the tree
  behind the GIL; retiring it is tracked separately. TLABs and per-thread
  store buffers were never built.
- *Contract restatement.* What incminimark actually relies on is (a)
  exclusive heap access during each collection **step**, (b) enumerability
  of all roots at that instant, (c) all inter-step mutations caught by the
  write barrier. Only (a) is what the two mechanisms above disagree about;
  (b) and (c) are the same either way. The audit that this holds for the
  *incremental* major is source-verified: major slices already execute
  at minor-collection time (`minor_collection_with_major_progress`,
  incminimark.py:824), hence inside STW windows; the barrier is deliberately
  newvalue-agnostic ("the incremental GC nowadays relies on this fact",
  incminimark.py:1516-1518) and its records are consumed at the next minor
  (VISITED clear + `more_objects_to_trace` re-append, incminimark.py:
  2079-2083). Concurrency therefore lives entirely in whatever supplies (a),
  and the ported collection algorithms are not rewritten either way.
- *Moving nursery is kept.* JIT inline nursery allocation is a performance
  pillar; embedder-boundary pointer stability is solved by pinning (upstream
  already has it: `GCFLAG_PINNED`, incminimark.py:148) or oldgen promotion,
  not by adopting a non-moving allocator.
- *Deviation zones are exactly four*, and everything else remains parity
  (N2): (1) allocation front-end, single bump pointer → per-thread TLAB
  chunks carved from the one nursery region (per-chunk pinning walls);
  (2) the barrier **producer** side — atomic header-flag RMW, per-thread
  SSBs flushing into the canonical `old_objects_pointing_to_young`, so the
  consumer code stays a line-by-line port; (3) codegen addressing:
  `nursery_free` baked static address → thread-context-relative, with the
  inline fast-path shape unchanged; (4) the safepoint subsystem itself,
  which has no upstream counterpart (and therefore no merge surface).
- *Non-options, with their failure mode*:
  per-access locks on the GC handle (fixes the data race, not the
  moving-collector root-visibility race — demonstrated by the gh#396
  cargo-test heap corruption); interior mutability without synchronization
  (hides the UB); per-thread heaps (breaks shared-heap semantics, and
  process-global caches already made it flaky in practice); a concurrent
  non-STW collector (requires rewriting incminimark's tracing/evacuation).
- *TLS discipline refined*: thread-local state is legitimate exactly where
  the design is per-thread (TLABs, mutator contexts); what is forbidden is
  a TLS raw pointer into unsynchronized shared mutable state — the gh#396
  defect. Staged execution plan (P0 soundness core → P1 TLAB/SSB/back-edge
  polls → P2 `_thread` + object-model epic) is recorded on gh#396.

### 3.4 Backends: three thin backends, not six hand-written

D08.2 closed with a research question: can the meta-JIT sit on a lower-level
code generator that owns regalloc/instruction selection? RPython answered
"no" by default and paid ~300k LOC of hand-written backends. pyre's answer is
to keep **three thin backends behind one trait** rather than owning six full
instruction sets. The current primary is **dynasm** — direct machine-code
emission via dynasm-rs, favored for compile latency and fine control;
**Cranelift** is the portable option that does take regalloc/instruction
selection downward (the literal "yes" to D08.2's question); **wasm** is a
target of its own. The trade is explicit: a less specialized lowest layer in
exchange for not owning six instruction sets. When a backend is the
bottleneck (compile latency, missing patterns), the recourse ladder is: fix
usage → dynasm fast path → upstream Cranelift work — not a hand-written pyre
backend.

### 3.5 What is explicitly *not* on the map (anti-roadmap)

Each item below was built by PyPy, measured, and deleted or abandoned —
the reports plus later history are the evidence (assessment §§2–3). Do not
rebuild them in pyre/majit without new, written justification that addresses
why the original failed:

- **Pluggable object spaces** (thunk/taint/logic/proxy spaces). The objspace
  *interface* stays; runtime-swappable semantics died — JIT-hostile,
  dispatch-costly. Security = rsandbox at the translation layer instead.
- **Multimethod dispatch + table slicing** in the object space.
- **Two-modules-per-type** layout; EU-era std-objspace shapes generally.
- **High-level backends / ootype** (JVM/CLI/JS). pyre-wasm is not this: it
  is a low-level target through the same lltype-like pipeline.
- **CPS/graph-transform stackless.** If deep coroutine support is ever
  needed, follow modern PyPy (stacklets/continulets — small per-platform
  assembly), not D07.1's transform. The "views" composability analysis
  remains valid input for any future design.
- **rctypes-style FFI** ("write extensions in RPython/Rust and translate
  them everywhere"). The C-extension story must follow the approaches that
  actually worked (cpyext-class emulation / HPy / cffi-class FFI) — §5,
  Phase C.
- **Exotic default representations** (ropes, string slices, prebuilt int
  boxes, optimal arrays). Adaptive **storage strategies** — the shape that
  shipped — are the porting target.
- **Syntax extensibility** as a goal. pyre tracks CPython exactly; the
  compiler front-end is not an extension point.
- **Naive refcounting**, conservative GC as default, or any GC not derived
  from the incminimark lineage.

**Anti-roadmap vs. provisional alternatives — a distinction.** The
exclusions above concern *re-adopting a mechanism PyPy abandoned* (ropes,
multimethods, ootype…) as a default; reversing one requires the §6
justification. They say nothing about pyre's own **provisional
implementations** of the *winning* mechanisms. Where the faithful port is not
yet in place, a simpler alternative is an acceptable **interim** — provided
the orthodox port stays the target (A6/N2), the deviation is tracked
(A7, §3.1), and it remains **reversible to the canonical implementation**.
Shipping an alternative now never forecloses correcting to the orthodox one
when measurement or need calls for it; that correction is expected, not a
policy change.

### 3.6 Configuration: small supported matrix, build-time resolution

D13.1's mechanism (options resolved at translation time, dependency-checked,
write-once) maps to Cargo features + build-time codegen; its *lesson* maps to
policy: the 1.0 compatibility matrix showed "any combination of aspects"
failing in practice, and post-EU PyPy pruned to essentially one supported
configuration. pyre keeps:

- **One blessed default configuration** that is always green and always
  benchmarked (this is what CI and check.py gate).
- Experimental behavior behind `PYRE_*` / `MAJIT_*` env gates or features.
  Default-off is the norm; a gate may be default-**ON** where its reason for
  existing is to keep the switched-off arm reachable as a one-binary A/B, and
  then its registry row has to name the trigger that retires it. Either
  polarity carries an owner-issue and an intended flip-or-delete decision. A
  gate is a staging area, not a home. The registries are `pyre/gate-triage.md`
  and `majit/gate-triage.md`; a gate read from the environment with no row
  there fails `gate_triage_complete`.
- Aspect combinations (e.g. wasm × JIT × GC modes) are individually declared
  supported or unsupported; silence means unsupported.

---

### 3.7 The portal boundary: warmspot split in two

Upstream mints the portal at translation time. `warmspot.apply_jit` — the body
of the translator task literally named "JIT compiler generation"
(`task_pyjitpl_lltype`) — derives each driver's green/red specification
(`make_args_specification`), rewrites the `jit_merge_point` and `can_enter_jit`
markers into calls (`rewrite_jit_merge_point`, `rewrite_can_enter_jits`), and
fills the fields `JitDriverStaticData` declares but never computes. Upstream's
`jitdriver.py` is an attribute container with two executable statements
precisely because warmspot writes the rest. pyre splits that pipeline across two
layers, unevenly:

- **The derivation and the marker erasure do run at build time**, over Charon
  LLBC, in majit-translate's `jtransform` and `CallControl::setup_jitdriver`,
  driven from `pyre-jit-trace/build.rs`. The derived green/red layout is
  asserted against the real MIR operands and a mismatch fails the build. This
  half is at the right layer and is not debt.
- **`apply_jit` itself is unported.** `task_pyjitpl_lltype` assembles every
  upstream-shaped argument and then returns `TaskError`, because majit-translate
  does not depend on majit-metainterp; `warmspot.rs` is a `pub use` namespace,
  not an implementation. Seven `missing_task_leaf` sites exist across that
  driver, so the stub is not unique — it is named here because the fields it
  would fill are instead written by hand from consumer source.
- **A second codewriter runs at runtime.** majit-translate's
  `transform_graph_to_jitcode` consumes a `FunctionGraph` once per build;
  pyre-jit's `transform_graph_to_jitcode` consumes a user `CodeObject`, is
  fallible, and runs unboundedly. Upstream has one, over the interpreter's own
  graphs. This is the A1 debt in this area — it is written, it carries Python
  opcode semantics, and it has already produced a wrong answer of exactly the
  class N3 names: in a chained blackhole resume `portal_frame_reg` aliased the
  caller frame, so an inlined callee's `LOAD_GLOBAL` indexed the caller's
  `names` table. A1 is **not** weakened to accommodate it; it stands as a
  tracked generation defect whose convergence target is majit-translate's
  codewriter.

**Measured cost, re-measured 2026-08-26** (dynasm release binary, best-of-3
`time.process_time()`, one million iterations, both loop shapes run against
CPython 3.14 and PyPy 7.3.20 on the same machine). The cliff moved rather than
vanished. On a **call-free** hot loop pyre now pays nothing for either hook —
0.0034 s profiled against 0.0047 s bare. On a loop that **calls** a small
function every iteration it pays two orders of magnitude: 272–328× for
`sys.setprofile` and 178–215× for `sys.settrace` across runs, where CPython
pays 2.6–3.3× and PyPy pays about 1× and stays compiled. The charge is
therefore per *call event*, not per profiled frame, and the earlier
1168–2836× figure is superseded — it was taken before the bracket below
landed.

Event counts still match, which is what licenses the comparison: at a million
iterations all three runtimes report the same `call` / `return` / `c_call`
multiset. pyre additionally reports 28 `importlib._bootstrap` weakref-callback
frames that reference counting had already reclaimed before the profiled
window opened — a collector-timing difference outside the loop body, 0.003% of
the events, and not JIT-attributable, since it survives `PYRE_NO_JIT=1`.

**The missing bracket is restored, and the falsification below came back
clean.** Upstream brackets the portal itself: `PyFrame.execute_frame` wraps
`dispatch` — the function that carries the merge point and nothing else — in
`ExecutionContext.enter`, `call_trace`, then `return_trace` and `leave` in
`finally` clauses. pyre had put that bracket *inside* the plain dispatch body
(`eval_frame_plain_with_resume`) and left the JIT dispatch body bare. It no
longer does: `eval_with_jit_inner` now calls `call_trace` and `return_trace`
around the portal, and `frame_tracing_active` — the gate that had been sending
every traced or profiled frame down `execute_frame_plain` — is narrowed to the
per-frame `!frame.get_w_f_trace().is_null()`. The global `profilefunc` and
tracefunc disjuncts are gone and no events went missing, which is exactly what
this entry predicted. **The gate is no longer the implementation of frame
events for JIT-eligible frames**, and a profiled call-free loop now measures
the JIT.

**The two hooks now fail for different reasons, and neither is the bracket.**
Under `MAJIT_STATS=1` on the call-bearing loop:

- `sys.setprofile` **loses the compilation**: `loops_compiled` falls 4 → 1 and
  `loops_aborted` rises 0 → 5. The five are `abrt_bridge`, but the three
  `giveup_*` splits that decompose that slot all read 0 and
  `abrt_unclassified_default` reads 5, so no bridge is involved — these are
  the aborts that stage no reason at all, which is the case that slot exists to
  separate. `MAJIT_LOG` puts them one line after `init-sym` with no recorded
  op in between: the callee compiles, the caller never starts. After the fifth
  the green key is banned (`abort_ceiling_banned=1`) and the next 174,776
  attempts are refused outright, so the loop is interpreted for the rest of the
  run.

  The abort has a location even though it has no name. A breakpoint on
  `MetaInterp::abort_trace` never fires, so these five do not come from it;
  they come from the other site that bumps the same slot, `jitdriver.rs`'s
  reason ladder inside `merge_point`, which does the live-cleanup and
  accounting halves itself. Its backtrace runs through the *callee's* own
  activation — `call_user_function_with_ctx` → `eval_with_jit` →
  `eval_loop_jit` → `maybe_compile_and_run` → `bound_reached` →
  `compile_and_run_once` → `merge_point` — so the outer loop's trace is
  discarded from inside the residual call, at the point the callee's own entry
  threshold fires, and not at the outer loop's merge point. Upstream carries a
  `Counters.ABORT_*` on every `SwitchToBlackhole`, so an abort that stages no
  reason has no counterpart there at all; staging one here is the first
  concrete task, and it is parity work rather than new diagnostics.
- `sys.settrace` **keeps it**: `loops_compiled=4`, `loops_aborted=0`,
  `guard_failures=3`, `back_edge_polls=0` — the whole summary line is
  byte-identical to the bare arm's.
  Nothing is lost to compilation or deoptimisation, so the whole 178–215× is
  executed per-call event dispatch inside compiled code.

Two mechanisms, one shared shape: the per-call profile/trace dispatch is not
folded. That is the work this entry now tracks.

**What upstream does not do.** It does not fold the tracing state away.
`ExecutionContext` declares `_immutable_fields_` with `profilefunc?` and
`w_tracefunc?`, yet the recorded traces read both as ordinary fields and guard
them, and the comment directly above that declaration says so: the fields
"should be known to a constant … but they're not". They are cheap because
they sit on the entry bridge, once per frame activation — not because they
disappear. Nor would the declaration help here: `quasi_immut_descr` requires a
constant struct operand, and pyre's `ec` is a portal red (`PYPYJIT_RED_VARS`),
so it is never one. The per-opcode half is a different mechanism again —
`dispatch_bytecode`'s explicit `we_are_jitted()` arm tests the *per-frame*
`w_f_trace` through the virtualizable `debugdata`, not the global tracefunc.

Where the green does pay is the profiled-call dispatch: `call_valuestack` and
its keyword/ex siblings branch on `get_is_being_profiled()` before
`call_args_and_c_profile`, and a real green folds those branches to nothing in
the unprofiled trace while giving the profiled state its own cell, counter and
procedure token. With the bracket landed and the call-free shape flat, this is
now the remaining step rather than the last one, and the measurement above says
which half to take first: `settrace` keeps its compilation, so its cost is
entirely in that unfolded branch; `setprofile` aborts the recording before the
branch can matter.

**Falsification, run 2026-08-26 — passed.** The prediction was that restoring
the activation bracket would leave event counts unchanged with the gate still
in place, and would then let the gate's `profilefunc`/global-tracefunc
disjuncts be dropped without losing events. Both held: the gate now reads only
the per-frame `w_f_trace` and the `call`/`return`/`c_call` multisets are
unchanged, so the bracket was what the gate was standing in for.

The successor claim, which this entry now stands or falls by: the residual
cliff is the unfolded per-call dispatch and nothing else. It is wrong if
folding `get_is_being_profiled()` out of the unprofiled trace leaves the
call-bearing loop more than an order of magnitude off its bare arm, or if the
`setprofile` aborts survive a recorder that admits the profiled call — either
would mean a third mechanism is in play that neither the bracket nor the green
accounts for.

---

## 4. Norms (operating rules)

**N1 — Layering.** majit never depends on pyre. pyre-interpreter stays
traceable "straightforward Rust"; when majit-translate cannot lower an
interpreter construct, the default resolution is a majit improvement, the
fallback is a *tracked* residual call, and the forbidden resolution is
contorting interpreter semantics to please the translator. "Rust can't be
meta-traced" is never a valid excuse (AGENTS.md).

**N2 — Parity.** Line-by-line structural parity with modern PyPy/RPython:
same modules, names, data structures. No Rust-native collection where
RPython used an attribute/forwarded slot; borrow-checker workarounds minimal
and documented with the RPython original cited. Do not delete RPython
methods to "simplify". (Full rules: AGENTS.md; they are part of this
charter.)

**N3 — Frame identity.** One red frame per interpreted frame, everywhere:
tracing (MIFrame), resume, blackhole, bridges. Collapsing inlined callees
onto shared anchors is the known root cause of a whole bug class
(LOAD_GLOBAL namespace confusion, pycode miscompiles). The reports' own
virtualizable design assumed per-frame identity; RPython's 1-red-arg frame
shape is the convergence target.

**N4 — Correctness gates.** `cargo check` + `cargo test` (both feature
configs) green before commit; full benchmark suite (all 8) after JIT
changes with no regressions; bit-exact CPython 3.14 parity for observable
behavior; compliance-suite pass rate only moves up. Root-cause fixes only —
no workaround modules, no tolerances.

**N5 — Empirical flips.** An optimization becomes default-ON only with:
benchmark evidence on the suite (not one kernel — the 1.47×-gcc lesson),
green tests, and a kill switch that stays for one stabilization period.
Record refutations in memory/docs; D06.1-style negative results are a
deliverable, not an embarrassment.

**N6 — Translation latency is a first-class cost.** Whole-program
translation friction taxed PyPy for two decades (2h builds in D09.1; "the
attention of another dev for the whole sprint" culture). Guard pyre's loop:
incremental Rust builds, LLBC fingerprint skipping, prepass performance,
and no O(n²) in trace/compile paths (the #345 class). Wall-clock of the
edit-test cycle is reviewed like a benchmark.

**N7 — Documentation of thinking, not just code.** The EU reports' lasting
value is that the *reasoning* was written down, which is what made later
deletion rational rather than amnesiac. pyre's analogs — issue epics, memory
files, this charter — must record why, what was measured, and what would
falsify the decision. A major mechanism landing without a written rationale
is incomplete.

---

## 5. Macro-strategy: the ten-year arc

The EU project's honest final ledger — research vision delivered, product
usability consciously deferred, and the deferral costing a decade — defines
pyre's sequencing. The phases overlap; the ordering states *priority under
contention*, not a waterfall.

**Phase A — The JIT spine (now).** Make meta-tracing-by-translation
boringly correct and PyPy-fast on the benchmark suite. Concretely: eliminate
compilation cliffs (unported opcodes/`abort_permanent`, no-token loops,
recursion walls), converge frame/resume machinery on RPython shapes (frame
identity epics, pc_map retirement, resume rebuild parity), and close the
fib_recursive-class call-frame gap. Exit criterion: parity-class performance
with PyPy on the suite with the JIT never producing wrong answers. This
phase outranks everything, because it is the part the reports proved *can*
fail structurally (the 2007 JIT) and the part every later phase builds on.

**Phase B — Language and stdlib completeness.** CPython 3.14 compliance ramp
(regrtest enablement, test-infra rounds, GC-root classes of bugs), full
built-in coverage, generators/async, memory-model soundness under the
compliance suites. This is the "engineering after research" that PyPy
deferred; pyre schedules it as a standing workstream that grows as Phase A
stabilizes, not as an afterthought.

**Phase C — Adoption surfaces.** The two things history says decide real
use: **C-extension compatibility** (choose and execute an HPy/cpyext-class
strategy — with the rctypes failure and the cpyext cost curve as priors;
target a decision document, then years of grind) and **no-GIL parallelism
actually delivered** (thread scheduling, concurrent GC hardening, and the
free-threading ecosystem story that CPython 3.13+ opened). no-GIL is pyre's
principal differentiator against both CPython and PyPy; it must land as an
aspect-layer property (§3.3), never as interpreter-source complexity.

**Phase D — Platform axes.** The validated axes of l×o×p, modernized:
**wasm** as the p-axis (browser/edge/embedded — D11.1's reopened questions:
size budgets, reduced builds, sandbox-by-substrate), **rsandbox** as the
o-axis security aspect, cross-platform backend maturity (dynasm/AArch64/x86
via cranelift). Each platform combination enters the supported matrix
explicitly (§3.6) with its own CI, or stays experimental.

**Phase E — majit as a public framework.** The l-axis: majit as *the* way to
give a Rust interpreter a tracing JIT — the role RPython proved with
Pyrolog/Topaz/HippyVM. Requires: API stability, documentation of the hint
vocabulary, at least two non-pyre consumers in CI (aheui-mjit today), and
the discipline that pyre-specific needs land as general mechanisms or not at
all. This is last not because it matters least but because a framework
extracted from one working product beats a framework designed for
hypothetical ones — also an EU-report lesson ("RPython was 'just' the
implementation language" until it was proven).

**Standing constraints across all phases**: N1–N7; the anti-roadmap (§3.5);
and the A7 duty to delete. Every phase inherits the reports' deepest single
lesson: *the architecture survives because the interpreter stays a clean
executable specification and everything else is generated, woven, measured —
and replaceable.*

---

## 6. Amending this charter

This charter changes by evidence, in writing: a proposal must cite the
measurement or upstream-PyPy precedent that motivates it, state which axiom
or norm it modifies, and record what was tried before. Additions to the
anti-roadmap require the failure evidence; removals from it require new
justification addressing the original failure. The document history is part
of the document.
