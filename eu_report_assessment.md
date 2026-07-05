# Assessment: what in the EU reports is still right, and what is not

Companion to `eu_report.md` (the objective digest — read it first). That file
contains only what the 2004–2007 documents say; **this** file judges those
claims against two later bodies of evidence: (a) what PyPy itself did in the
following ~18 years (the current source tree under `pypy/` and `rpython/` in
this repository is the ground truth), and (b) what pyre/majit has already
reproduced or contradicted. The normative conclusions for pyre are drawn in
`pyre/design.md`; this file is the evidence table.

Legend: **VALIDATED** — held up as stated; **SUPERSEDED** — the implementation
was replaced but the underlying concept survived (usually in changed form);
**REFUTED** — abandoned by PyPy itself or contradicted by later evidence;
**OPEN** — still genuinely undecided.

---

## 1. The big bets

| EU-report claim | Verdict | What actually happened |
|---|---|---|
| The interpreter is "an executable specification"; the JIT must be *generated* from it, never hand-written | **VALIDATED** | The single most durable idea in the corpus. The tracing JIT (2008+) is a different *generator*, but the principle — performance orthogonal to interpreter evolution — is exactly what let PyPy track 2.7→3.x with one JIT. This is the founding axiom of majit. |
| The specific JIT generator: binding-time analysis + timeshifting (offline partial evaluation) | **REFUTED** | Abandoned within ~2 years of the final report, replaced by the meta-tracing JIT (pyjitpl5). The reports' own §3.7 "open issues" (eager-branch code explosion, non-termination, unrefined widening, no hotspot detection) list precisely the reasons. The authors' hedge — "the best solution is probably something in between these extremes [eager PE vs Psyco-lazy]" — was prophetic: tracing *is* the lazy extreme, made practical. |
| Promotion, red/green binding times, virtuals, virtualizables, deep-freeze/immutability hints | **VALIDATED** | Every one of these concepts survived the PE→tracing rewrite and is load-bearing in `rpython/jit/` today: `promote()`, greens/reds on JitDriver, virtuals in the optimizer, `_virtualizable_` frames, `@elidable`/quasi-immutable fields. Architecture insight outlived implementation strategy. |
| "Dynamic analysis is ultimately more powerful [than static]" (D05.1 §8.1) | **VALIDATED** | By PyPy's own trajectory: the static PE JIT lost to runtime tracing; static type inference (annotator) survived only as an offline build step, never as the optimization engine. |
| Manual JIT invocation (`pypyjit.enable`) acceptable, hotspot detection "not worked on yet" | **REFUTED** | Tracing JIT made counter-based hotspot detection (`jit_merge_point`/`can_enter_jit` thresholds) fundamental. No modern deployment exposes manual enabling as the primary interface. |
| "By construction, the JIT should work correctly on absolutely any kind of Python code" (frame introspection, `sys._getframe…`) | **VALIDATED** | Survived as PyPy's core correctness discipline: the JIT falls back to the interpreter (blackhole/deopt) rather than restricting the language. Directly encoded in pyre's AGENTS.md frame-identity rule: JIT/interpreter divergence is a *generation defect*, never an accepted limitation. |
| 1.47× unoptimized gcc on arithmetic ⇒ "abstraction overhead has been correctly removed" | Technically true, **misleading in scope** | The result was real but held only for the integer-arithmetic kernel with hand-placed hints. General speedups arrived only with tracing (post-project), exactly as the report's fine print admitted ("more extensive hints are necessary … after the project"). Lesson: kernel benchmarks validate machinery, not the product. |

## 2. Interpreter architecture

| Claim | Verdict | Notes |
|---|---|---|
| Bytecode-interpreter / object-space separation with wrapped black-box objects | **VALIDATED** (as an internal layering) | Modern PyPy and pyre both keep interpreter vs `objspace/std` (pyre-interpreter vs pyre-object). The W_Root/`w_` discipline, `OperationError`, gateway/`unwrap_spec` machinery all survive nearly verbatim. |
| Objspace as a *pluggable* runtime extension point (thunk, taint, logic, proxying spaces) | **REFUTED** | Every alternative object space was deleted. Proxying spaces multiply dispatch cost and are hostile to the JIT (every operation must be traceable through the proxy). The flow object space was also divorced from the objspace interface (`rpython/flowspace/`). What survived is the *interface*, not the pluggability. |
| Multimethod dispatch + table slicing in StdObjSpace | **REFUTED** | Modern PyPy removed the multimethod machinery; operations are plain methods/descriptors on W_ classes. The EU-era justification ("can probably be translated to … efficient multimethod code") never paid off; the JIT made dispatch cost irrelevant and the machinery pure complexity. pyre correctly ports the *modern* shape. |
| Two-modules-per-type layout (`xxxtype.py` + `xxxobject.py`) | **REFUTED** | Merged into single `xxxobject.py` files in modern PyPy. |
| Multiple hidden implementations of one user-visible type | **VALIDATED — in changed form** | The multidict/multilist idea matured into **storage strategies** (list/dict/set strategies, mapdicts, celldicts) — adaptive per-instance representation, chosen at runtime, JIT-visible. The exotic static variants (ropes, string slices, optimal arrays) died exactly as D06.1's own measurements predicted. |
| "Wrap everything, including ints; unboxing is a translation-time optimization" | **VALIDATED** with a caveat | The clean-source/optimize-later doctrine held. But the promised optimization (tagged pointers) stayed off by default in PyPy; what actually removed boxing cost was the JIT (virtuals). pyre note: pyre's `tagged-int.plan.md` revisits this with different economics (Rust enums/NaN-boxing options); the EU-era measurement (+7% at best, only with const-folding) is a sober prior. |
| Frame classes specialized/generated at bootstrap; argument parsing via generated code | **SUPERSEDED** | The mechanism (metaprogramming at bootstrap, invisible to the annotator) is RPython-specific. The need it served — keeping the analyzable code static — is served in pyre by Rust's own static types and proc macros. |
| Parser/compiler: flexible grammar-from-data, AST-direct building | **SUPERSEDED** | The three-generation pivot story is a good case study in porting-vs-rewriting, but syntax-extensibility as a goal (Oz-syntax, `import csp_syntax`, AOP hooks) died. pyre uses RustPython's compiler front-end targeting CPython 3.14 bytecode — the "track CPython exactly" pole won over the "extensible syntax" pole. |

## 3. Translation and aspects

| Claim | Verdict | Notes |
|---|---|---|
| Whole-program type inference over a live program image; RPython deliberately unspecified | **SUPERSEDED** for pyre; costly even for PyPy | It worked, but the costs the reports admit (slow, non-incremental, first-error-only, cryptic diagnostics — the entire reason RPylint existed) were never fixed; "modular annotation" (D05.1's stated future need) never happened. Rust + Charon LLBC extraction gives pyre the same whole-program low-level view with a real type system, incremental builds, and real error messages. The annotator survives in majit only as the *binding-time/representation* analysis it also was in RPython's JIT, not as type recovery. |
| Translation aspects: GC, threading model, stackless, sandbox woven at translation time, invisible in interpreter source | **VALIDATED** — the second most durable idea | PyPy's GC transformer, sandbox transform, and (while it lived) stackless transform all confirmed it. pyre already re-instantiates it: majit-gc weaving/write-barrier hooks, rsandbox as a compile-time aspect (PR#304), JIT generation itself as "just another aspect". The refutation of *manual* refcounting sprinkled through CPython sources is as true in 2026 as in 2005 — and is now also the free-threading argument. |
| "Any combination of aspects can be selected freely" | **PARTLY REFUTED** | D13.1's own compatibility matrix already showed the combinatorial claim failing (JIT = default config only; logic space = framework GC + stackless only; tagged ints = Boehm only). Post-EU PyPy pruned the matrix instead of completing it: one GC, one backend, one objspace. Real lesson: aspects are valuable, but each supported *combination* is a product you must test; keep the matrix small. |
| Refcounting "a possibly viable option" worth optimizing | **REFUTED** | 2× slower naive, no cycle collection; deleted. Exact generational moving GC won (minimark → incminimark), fulfilling D06.1's "biggest single improvement would likely be … a more sophisticated garbage collector". majit-gc (nursery+oldgen+incremental+card marking) is the port of the winner, not the contenders. |
| GC-in-the-implementation-language + memory simulator for testing | **VALIDATED** | incminimark is RPython; testable on the simulator. majit-gc being Rust-on-Rust with test harnesses is the same move. |
| Root finding "one of the hardest problems"; future idea: let the JIT find roots since it knows frame layout | **VALIDATED** | Shadow-stack won for interpreter code (asmgcc was tried and removed); JIT-known frame layout became GC-traced jitframes. pyre's wasm Option A GC-traced JITFRAME and the PyFrame→W_Root work (#355) are the same resolution. |
| Stackless transform: portable CPS-style unwinding, "C stack as cache", +17–28% cost | **REFUTED** as shipped | PyPy retired the graph transform and adopted stacklets/continulets — small per-platform assembly stack switching, i.e. the approach the reports had positioned as the fallback. The composability ("views") analysis and coroutine-pickling semantics remain intellectually valid; the mechanism does not. pyre should not resurrect the transform. |
| Extension modules in RPython + rctypes; "one source fits all Python implementations" | **REFUTED** | rctypes died (its own report documents the pivot mid-flight and the moving-GC blockage); the extension compiler never became usable. What worked later: cpyext (C-API emulation) and cffi. The strategic error the final report admits — deferring extension-module engineering — cost PyPy a decade of adoption. Direct warning label for pyre's HPy/ABI roadmap item. |
| ootype + CLI/JVM/JS backends; "p" axis of l×o×p | **REFUTED** | ootype and all high-level backends were deleted from RPython (~2012-13). The C backend is the only survivor. The "l" axis, by contrast, was genuinely validated post-project (Pyrolog, Topaz, HippyVM, Pycket, RSqueak) — the framework *is* language-generic. For pyre: majit's multi-consumer design (pyre, aheui-mjit, toy interpreters) inherits the validated axis; backend proliferation should follow cranelift/dynasm/wasm need, not platform evangelism. |
| Config system: translation-time resolution, write-once options, dependency checking | **VALIDATED** | Still how RPython builds work. pyre's analog is Cargo features + build.rs codegen for build-time choices and `PYRE_*` env gates for runtime experiment flags; the discipline worth keeping is D13.1's: declared dependencies between options, and a small, explicitly supported matrix. |

## 4. Optimizations (D06.1) — the empirical record

- The **methodology** aged better than most results: benchmark suite spanning
  workload types, nightly regression tracking, per-optimization toggles,
  publishing negative results. This is check.py's lineage.
- **Individually validated**: method cache with type version tags (still in
  PyPy *and* CPython — the report predicted the port), CALL_METHOD-style
  bound-method elision (CPython 3.11 LOAD_METHOD/adaptive specializations are
  the same idea), key-sharing instance dicts (CPython 3.3 PEP 412 = SharedDict),
  string interning, exception-free internal interfaces.
- **Individually refuted**: prebuilt int boxes (cache-hostile), ropes/string
  slices/multilists as defaults, small-dict linear scan, bytecode
  type-special-casing (the JIT does it better), CALL_LIKELY_BUILTIN (dropped
  once the JIT subsumed it).
- The **meta-lesson stands**: representation cleverness that fights the host
  hardware (extra indirection, cache misses) or assumes untuned application
  code loses on real programs. Measure; keep a kill switch; expect "no clear
  tendency" as the usual outcome.

## 5. Process

| Claim | Verdict | Notes |
|---|---|---|
| TDD as the primary QA instrument; tests instead of proofs; a test per feature and per bug | **VALIDATED** | Also how PyPy still works, and how pyre works (cargo test ×2 feature configs, compliance suites, `check.py` gate). |
| Multi-level test taxonomy (interp-level / app-level / compliance / translation) | **VALIDATED** | Maps 1:1 onto pyre: Rust unit tests / extra_tests + cpython_tests / lib-python compliance / majit-translate prepass+examples. |
| Distributed testing because translation tests dominate wall-clock | **VALIDATED** as a warning | Whole-program translation latency (~2h for the logic build in D09.1) was a permanent tax on PyPy development. pyre's equivalents (LLBC re-extraction minutes, prepass rebuilds) need the same active management (fingerprint skipping already exists). |
| Sprints/community process | Context-bound | Historically interesting; not normative for pyre. The transferable kernel: releases gated on green suites + docs, continuous trunk consistency, "no formal task distribution" self-organization. |

## 6. Still-open questions the reports flagged

- **Merge/widening policy** ("merging too eagerly may loose important
  precision and not merging eagerly enough may create too many redundant
  residual code paths") — the tracing JIT re-encounters this as trace scope,
  retracing, and bridge-vs-loop decisions. pyre is living inside this problem
  today (#345 compile bottlenecks, cross-loop-cut, bridge chaining).
- **Promotion state retention** ("can never be reclaimed") — still true in
  spirit: guard_value bridges and cached greens accumulate; PyPy manages it
  with trace limits/retrace counts rather than solving it.
- **Layering a meta-JIT on a lower-level JIT/VM** (D08.2 §3.7's research
  question) — pyre answers a variant daily: majit on cranelift, majit-on-wasmi,
  wasm backend. The question of how much to delegate downward (regalloc,
  inlining) vs control directly is still live engineering, not settled theory.
- **Reduced/"micro" interpreter builds** (D11.1) — never built by PyPy;
  pyre-wasm's size/coverage constraints re-open it.

## 7. Summary judgment

The corpus is two documents interleaved. One is a **strategy** — generate
everything from a single high-level executable specification; postpone
low-level decisions to translation; let tests, not proofs, carry correctness;
measure and keep the losers toggled off. That strategy is validated nearly
without exception, by PyPy's next two decades and by pyre's own results.
The other is a **portfolio of 2005–2007 implementations** of that strategy —
the PE JIT, multimethods, pluggable spaces, ootype backends, CPS stackless,
rctypes, refcounting — of which almost every item was later deleted *by the
same team*, usually for reasons their own reports already recorded as open
issues. The reports are therefore most valuable read as: (a) the canonical
statement of the strategy, (b) a catalog of measured dead ends that need not
be re-explored, and (c) proof that the team's willingness to delete its own
flagship implementations was itself part of the strategy.
