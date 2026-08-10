# PYRE_* gate triage

**Status**: living record. WS4 deliverable of `rework.md` (finding F5, gate
debt). Audited 2026-07-05 on branch `pc-map` by reading the actual
read-expression at every source site (four census passes). Polarity rule:
`is_some()`/`=="1"`/`.unwrap_or(false)` → default **OFF**; `is_none()` /
`!= Ok("0")` / `.unwrap_or(true)` → default **ON**; a parse of
number/path/list/mode → **VALUE** (config, not a boolean gate).
Re-audited 2026-07-18 on branch `nbody`: §1c added; 10 rows retired.

The charter (§3.6, A7) says a gate is a staging area, not a home: each live
default-ON experiment gate is kept only until its epic closes, then its OFF
path is deleted and the gate retired. This table is the standing list of what
to retire and when.

## Headline

The raw `rg 'PYRE_[A-Z0-9_]+'` count (~119) overstates the debt. **~20 of the
matches are not env gates at all** (Rust consts, macro-generated identifiers,
runtime symbols, or comment-only dead references). The 2026-07-05 audit found
that the **wasm trio** and the **#171 pair** were cleanly settled (epic closed
/ merged) and retired in that pass. The 2026-07-18 re-audit found no new live
default-ON gate that is safely retirable right now; §1c is book-keeping for
rows whose source readers were already deleted by closed epics after the
original audit.

## §1 — Retired this pass (5)

Hardwired ON. Behaviour is byte-identical (each was default-ON already; only
the opt-out capability is gone). The **wasm trio** removed the env read +
guest export + `set_*` + `AtomicBool` static; the constant-`true` reader fns
were then deleted outright and their call sites folded (including the dead
`is_loop` parameter of `build_wasm_module` this exposed);
verified compile-clean on native `majit-backend-wasm`, native
`pyre-jit`+`pyre-wasm-runner` (`--features dynasm`), and the wasm32
`pyre-wasm --features wasm-host` guest. The **#171 pair** inlined the
`&& enabled()` conjuncts and deleted the two helper fns; verified compile-clean
on `pyre-jit-trace --features dynasm` (exit 0; the `assembler.rs:1029` build.rs
panic in the log is the pre-existing stale-LLBC fail-open, not from this
change). #171 e2e is a JIT hot path — the full check.py suite should run before
the branch ships.

| var | feature | landed | why safe to retire |
|---|---|---|---|
| PYRE_WASM_CA | self-recursive CALL_ASSEMBLER guest→guest `call_indirect` arm | wasm campaign `654df9dd46`, suite 169/169 | wasm backend is separate; open wasm issues (#352/#262) are orthogonal to CA correctness |
| PYRE_WASM_ENABLE_BRIDGES | in-module inter-trace bridge chaining | same | same |
| PYRE_WASM_INLINE_ALLOC | inline nursery-bump alloc fast path | same | `gc_stress` still forces the helper-call path, so the stress override survives |
| PYRE_171_ORTHODOX | orthodox `w_list_append` charon-body descent (int-storage) | #171 PR#318/#322 MERGED | epic merged; user-approved retirement 2026-07-05 (overriding the standing user-curated note for this cleanup) |
| PYRE_171_OBJ_APPEND | orthodox descent for object-strategy lists | same | same; the `&& enabled()` conjuncts were inlined, helpers deleted |

## §1b — Default-OFF experiments retired 2026-07-05 (4)

Second pass over §5's "default-OFF experiments": each gate was judged against
the vendored RPython/PyPy source — is the ON-path a WIP parity port (keep) or
a pyre-invented mechanism that contradicts the PyPy design and can never
become default (delete the ON-path)? Four were removed (−299/+5 across 6
files); default behaviour is byte-identical since every deleted path was
opt-in dead code.

| var | ON-path | why permanently unlandable |
|---|---|---|
| PYRE_KEPT_OVERRIDE | `StackSource` bytecode-provenance lattice sourcing a kept stack slot from a local at bridge resume (~230 L, liveness.rs + state.rs consumer) | no PyPy analog — resume rebuilds the operand stack from resume-data boxes, never re-analyzes bytecode; the guard-half was already deleted as vstack-mirror-superseded in PR#292 (`910ffd4e64`), this was the orphaned bridge-half |
| PYRE_RELAX_124 | force-bypass of the two kept-stack branch-guard declines | known-unsound diagnostic: regressed 23/25→17/25 on the #124 corpus in an earlier retirement; the sanctioned route is the vstack mirror (#73/#423), under which the declines die naturally |
| PYRE_NO_DE | suppress single-pass direct entry, fall back to re-interpretation | W2-era diagnostic (W2 refuted); direct entry is the `ContinueRunningNormally` portal shape itself |
| PYRE_STRICT_TARGET_TO_PATH | audit probe disabling the cross-module leaf-match fallbacks in call-target→CallPath resolution (3 sites) | one-time #91 quantification sweep; development since has refined the fallback (suffix-carrier, alias-cluster dedup), i.e. the fallback is the accepted adaptation endpoint |

**Deferred, not retired** (active on other branches; touching them on pc-map
would only manufacture conflicts):

- **PYRE_P2_DRAIN** — RETIRED. The drain became the only bridge-carrier
  consumer once it compiled N-deep carriers (recipes 1..=7); the buggy
  framestack-walk escape hatch it gated was deleted. The branchy inlined-callee
  continuation gap's plain-branch subcase is closed unconditionally (the
  carrier-resume sub-walk int-binop inline landed without a gate in #748); its
  exception-unwind subcase is handled by the try-block callee inline +
  carrier-boundary raise delivery, now unconditional with the
  `PYRE_FBW_TRYBLOCK_INLINE` / `PYRE_FBW_CARRIER_RAISE` gates retired.

**Judged KEEP** (genuine WIP parity port): `PYRE_FBW_VABLE_SCALAR_CA` was kept
here as "S0 seam of the vable-owner rework toward `direct_assembler_call` scalar
args" — **that judgement was wrong and is reversed in §1d**: the 2026-07-25
parity pass read `direct_assembler_call` and found its ON design is what
upstream's `num_red_args` assert forbids. Retired.
**`PYRE_CARRIER_EXC_RESUME` — RETIRED 2026-08-06, ON path deleted.**  It was kept
here as "default-off; threads the guard-failure exception into the bridge sym for
the depth-2 carrier exception-resume slice #343/#126 — inert until validated",
with two parity gaps booked as pre-flip work, and was then upgraded to "a live
adoption target rather than an inert one" on the strength of §1e's reachability
census.  **That upgrade was wrong and is reversed here.**  §1e instruments
`handle_fail`, one layer above the seed site; the seed additionally required
`sym.current_exc_value.is_null()`, and — decisively — its write is discarded by
the walk-start seed.  The site is reachable AND the gate was inert.  Reachability
was never the question it failed.

The account this replaces ran: pyre's standing-exception maintenance for an
exception-guard bridge lives in `seed_bridge_standing_exception_from_current`
(`state.rs`), not gated, sourcing the exception from `sym.current_exc_value`
falling back to `get_current_exception()` — the *execution context's* current
exception, the `sys.exc_info()` mirror, a different slot with different lifetime
rules than upstream's `cpu.grab_exc_value(deadframe)` — so the gate's only effect
was to write `guard_exc` into `current_exc_value` beforehand and let the ungated
code pick it up.  **That was already only half true on the day it was written.**
`seed_standing_exception_for_walk` (`jitcode_dispatch/mod.rs`, called at walk
start from `bridge_subwalk.rs`) already had its present shape: it runs AFTER
`setup_bridge_sym` and sources from `BH_LAST_EXC_VALUE`, which
`trace_and_compile_from_bridge` (`call_jit.rs`) publishes from
`cpu.grab_exc_value`'s result on the exc-edge route, zeroes when the guard
carried no exception, and whose third combination (`pending_exc &&
!route_exc_edge`) declines before any walk.  For an exception-guard bridge that
function returns from one of its first two arms in every case — overwriting all
five exception slots on a non-null read, clearing them on a null one — so it
never reaches its own `last_exc_box` short-circuit on that flavour.  On the
single-frame walk the pre-seed's only possible effect was to write a pointer the
walk seed then rewrote, and the source divergence named above is closed.

Measured 2026-08-06 with a probe at BOTH seed sites across 369 synth benches: 23
exception-guard bridges in 14 benches, 7 of which the gate would have seeded, and
in all 23 the value at the walk seed equalled the value at the setup seed,
pointer for pointer (the non-seeding cases read `0x0` at both).
`PYRE_CARRIER_EXC_RESUME=1` over the whole `bench/synth` corpus is dynasm
386/386, byte-identical to gate-off, with zero jit-stats movement — agreeing with
the earlier corpus run under the gate (**dynasm 336/336 with the gate forced
on**, correctness results matching the default run, and the seven live-exception
producers of §1e among them, despite the seed site being entered 170 times).

"A green corpus under the gate is not evidence about the gate" still stands, and
it is why the 2026-08-06 evidence is the seed-site probe rather than the corpus:
the corpus could only ever show the no-op, never its cause.  It also revises the
cause inferred here.  The `is_null` conjunct was read as suppressing the
injection exactly when the EC already holds an exception; 7 of the 23 were not
suppressed at all, and the value they would have seeded is the value the walk
seed applies regardless.  Inert by redundancy, not by suppression.

**Scope of that redundancy, and what outlives the gate.**  It is a property of
the `dispatch_via_miframe` leg.  The multi-frame carrier leg does NOT run the
walk seed: `setup_bridge_sym` installs the inline carrier whenever
`resume_data.frames.len() > 1`, `trace_bytecode` returns through
`drive_bridge_carrier_walk` before the full-body-walk leg, and
`drive_bridge_frame_subwalk` seeds its sub-walk's `current_exception_seed` and
`class_of_last_exc_is_const` straight off `root_sym.last_exc_box()` — i.e. off
`seed_bridge_standing_exception_from_current`, with no `BH_LAST_EXC_VALUE` reader
anywhere on that leg.  So for a multi-frame exception-guard bridge (only the
`unwind_to_live_frame` shape survives `call_jit.rs`'s pre-walk decline) both
original complaints are still live: the `sys.exc_info()` mirror as the source,
and the early return when `last_exc_box` is already set, neither of which
`_prepare_exception_resumption` has.  That combination is unexercised by
`bench/synth` — the probe paired 23 for 23, so every exception-guard bridge in
the corpus took the single-frame leg — which is why the gate was never validated
and is why it is retired rather than flipped.  If the slice is built, its seed
must come from `BH_LAST_EXC_VALUE`, the `grab_exc_value` source, not from
`current_exc_value`.

The other pre-flip delta is **withdrawn as miscited**: it compared an
intermediate value to pyre's final one.  `prepare_resume_from_failure` calls
`execute_ll_raised` with the default `constant=False`, then
`handle_possible_exception` three lines later (pyjitpl.py:3169), which ends
`self.class_of_last_exc_is_const = True` (pyjitpl.py:3416).  Upstream's
post-resumption steady state is `True`, the same as pyre's.

## §1e — The grabbed guard exception is rooted for the whole handoff (2026-07-27)

`bridge_guard_exc` was booked as a pre-flip gap for `PYRE_CARRIER_EXC_RESUME`.
It is **not gate-specific**: the same grabbed pointer drives the default
blackhole resume, so the gate never bounded the exposure.  (That gate is retired
— §1b — and the `TraceCtx::bridge_guard_exc` carrier it was threaded through went
with it.  This rooting did not: its three parking sites are on the blackhole and
guard-failure paths, none of them the deleted one.)

`grab_exc_value` (`llmodel.py:240`) reads `jf_guard_exc` off the deadframe and
drops the jitframe, which was the collector's only handle on the exception
(`jitframe_trace`).  The handoff then decodes resume data and rebuilds virtuals
through the blackhole allocator before anything re-roots the value, so in that
window the exception — and the young `args` / `__traceback__` reachable only
through it — live behind a bare `i64` that a precise collector cannot see.
RPython's `grab_exc_value` result is a shadowstack-rooted local across the same
span.  Closed the same way as the six sibling raw-exception carriers
(`walk_jit_exc_value`, `walk_bh_last_exc_value`, …): `GuardExcRoot` parks the
value and a `GUARD_EXC_VALUE` root walker marks the carrier and forwards its
young children.  Parked at the three handoff owners — `handle_fail`,
`blackhole_resume_via_rd_numb` (which also covers the CALL_ASSEMBLER caller),
and `back_edge_internal`.

### Coverage census (339 files, `bench/` + `bench/synth/`)

Instrumenting `handle_fail` counted **732,660** guard failures:

| `guard_exc` | `is_guard_exc` | `should_bridge` | count |
|---|---|---|---|
| NULL | false | false | 648,725 |
| NULL | **true** | false | 48,327 |
| **NON-NULL** | **true** | false | **34,620** |
| NULL | false | **true** | 579 |
| NULL | **true** | **true** | 239 |
| **NON-NULL** | **true** | **true** | **170** |

So the window is entered with a live exception **34,790** times, and the 170 in
the last row are the bridge-route guard failures this section is about — the
grabbed value `call_jit.rs` publishes into `BH_LAST_EXC_VALUE`.  This read
"reachable, not inert" for the `PYRE_CARRIER_EXC_RESUME` seed site; **that
inference was wrong**.  The instrumentation is in `handle_fail`, one layer above
that seed, which additionally required a null `current_exc_value` and whose write
the walk-start seed discards.  The site was reachable AND the gate inert — see
§1b, retired 2026-08-06.  Seven benches produce them:
`inline_subwalk_property_mutates` and
`inline_subwalk_mutating_residual_abort` (11,482 each),
`type_name_surrogate_reject` (9,462), `named_reraise_sibling_hot` (1,418),
`exc_mixed_classes_bridge_flavor` (410), `handler_reraise_second_exc` (400),
`sre_pattern_methods` (136).

★ **TRAP** — an earlier revision of this section reported "zero coverage" from a
sweep whose every invocation had silently failed: `timeout` does not exist on
macOS, so each run died with `command not found` and produced no lines.  The
control used to validate that sweep did not go through `timeout`, so it did not
catch it.  Use `perl -e 'alarm N; exec @ARGV' --` instead.

### The walker is not load-bearing on any measured workload

A `gc_stress` build under `MAJIT_GC_STRESS` (full collection at the start of
every allocation, so the window's blackhole-allocator calls all collect) was run
over the producers above with the walker registered and with it suppressed:
`handler_reraise_second_exc`, `exc_mixed_classes_bridge_flavor`,
`named_reraise_sibling_hot`, `sre_pattern_methods` and
`type_name_surrogate_reject` all **pass identically both ways**.

So this stays a **parity fix at a reachable site**, not a demonstrated bug fix.
The likely reason it cannot be discriminated: on the residual-raise path the
same exception is still parked in `BH_LAST_EXC_VALUE`, which `walk_bh_last_exc_value`
already roots, and nothing drains that cell before the handoff completes.  What
the new walker covers is the case where it *is* drained first — untested,
because no workload produces it.

★ **TRAP** — the first attempt at this A/B forced `try_gc_collect()` at
`handle_fail` entry instead of using `MAJIT_GC_STRESS`.  That aborts with
`GC BUG: invalid type_id … site=object_total_size` on ~8/12 runs **with the
walker on as well**, which reads like a second defect but is not: an arbitrary
program point is not a safepoint, and the same bench is clean under the real
allocation-driven stress.  Force collections through the GC's own stress hook,
never at a hand-picked instruction.

## §1c — Retired since the 2026-07-05 audit (11): reader already deleted by a closed epic

Book-keeping only: these OFF-paths were deleted in source by the cited epics
after the 2026-07-05 audit; this pass removes their stale registry rows. The
2026-07-18 re-audit verified 0 Rust source read sites for each gate.

| gate | reader deleted by | note |
|---|---|---|
| PYRE_57_INLINE_NEXT | PR#387 (`e18ec90cac1`); follow-up `c6cfcb758c2` retired the kill-switch | stale §4 row removed |
| PYRE_SINGLE_PASS | PR#427 (`57849b62664`) | stale §1b keep mention and §5 list entry removed |
| PYRE_AUTHORITATIVE | PR#427 (`57849b62664`) + PR#415 (`7e3db1cc490`) | stale §1b keep mention and §5 list entry removed; `PYRE_PROBE_AUTHORITATIVE` is separate and remains live |
| PYRE_INNER_CLOSE | PR#427 (`57849b62664`) | stale §1b keep mention and §5 list entry removed |
| PYRE_NO_INNER_CLOSE | PR#427 (`57849b62664`); issue #152 closed 2026-07-13 | stale §1b keep mention, §4 row, and §5 list entry removed |
| PYRE_P2_COMPILE | PR#607 (`e1c43d3ff08`); follow-up `ca2640e797b` removed the gate | stale §5 deferred entry removed |
| PYRE_P2_FRAMESTACK | PR#374 (`9a97c47f6e9`) | stale §5 deferred entry removed |
| PYRE_P2_FS_COMPILE | PR#374 (`9a97c47f6e9`) | stale §5 deferred entry removed |
| PYRE_P2_AUTHORITATIVE | reader gone; attribution #374 per re-audit | stale §5 deferred entry removed |
| PYRE_SAME_GREENKEY | PR#390 (`802b79ff8db`); follow-up `111bdb4eeb8` dropped the gate | stale §1b deferred mention and §5 list entry removed |
| PYRE_FBW_REC_UNROLL | PR#374 (`9a97c47f6e9`) deleted `fbw_unroll_bound()` | stale §5 config-switch entry removed 2026-08-08. The successor knob `PYRE_FBW_REC_UNROLL_DEPTH` was never listed here and its reader `fbw_max_rec_unroll_depth()` is gone too (PR#887, `e5546b2ed36`) — both names read from nothing |

## §1d — Parity verdicts for the default-OFF `PYRE_FBW_*` seams (2026-07-25)

The 2026-07-25 pass judged each remaining default-OFF seam by the §1b question —
**is the ON path a WIP parity port, or a pyre-invented mechanism that
contradicts the PyPy design?** — reading the vendored `rpython/` + `pypy/`
sources rather than the gates' own doc comments.  Performance was explicitly
not a criterion.  Verdicts, each adversarially re-checked against the cited
upstream lines:

| gate | orthodox side | outcome |
|---|---|---|
| PYRE_FBW_VABLE_SCALAR_CA | **OFF** | **RETIRED** — the ON design contradicts upstream |
| `_MULTIFRAME` (retired) | **ON** | **RETIRED** — reader and OFF path deleted after the default-ON verification; the adopt is now unconditional when the multi-frame latch conditions hold. §1d's one remaining wrong answer (a `sys._getframe` that is itself the escaping residual reading the caller frame) was closed by `walker_ec_enter` / `walker_ec_leave` publishing the callee frame at the inlined-call push |
| PYRE_FBW_CALLEE_VSTACK | NEITHER | keep OFF; see §5 |

The walker's default-ON `PYRE_FBW_*` cluster was retired separately in #757.

**`_VABLE_SCALAR_CA` — retired, ON path deleted.**  The gate proposed passing
the callee's loop-carried locals as *extra scalar* CALL_ASSEMBLER args plus a
`VableExpansion` mapping each to a callee jitframe slot.  Upstream forbids
exactly that: `direct_assembler_call` records the op with the target
jitdriver's red args and asserts `len(args) == targetjitdriver_sd.num_red_args`
(`rpython/jit/metainterp/pyjitpl.py:3620`), and the PyPy portal's reds are
`['frame', 'ec']` (`pypy/module/pypyjit/interp_jit.py:67`) — literally the
`[callee_frame, callee_ec]` pair the OFF path already emits.  Upstream's
direction of travel is the exact inverse of the gate's: the CALLEE unpacks the
virtualizable, via `patch_new_loop_to_load_virtualizable_fields`
(`rpython/jit/metainterp/compile.py:425-461`), which *truncates* the callee
loop to `inputargs[:num_red_args]` (`:432`) and prepends a GETFIELD_GC /
GETARRAYITEM_GC per field read off the vable red arg (`:433-457`).  Because
that loop head dereferences a real heap frame, forcing the still-virtual frame
at the call — the allocation plus one SETARRAYITEM_GC per known element the
gate wanted to elide — is the upstream op sequence, not a pyre decline.

The retirement is byte-identical: the ON emitter was scaffolding that produced
the same red-only CALL_ASSEMBLER as OFF, and the only
`call_assembler_with_vable_expansion` constructor was `#[cfg(test)]`.
A follow-up commit deletes the `VableExpansion` type itself (`majit-ir`), the
`CallAssemblerDescr` accessor and `..._with_vable`/`..._with_expansion`
constructors (`majit-metainterp`), and the consumer arms in both backends —
including `genop_call_assembler`'s expansion tail, which allocated a callee
jitframe with `libc::calloc` and materialised the mapped fields into its slots.
Nothing in that tail has an upstream counterpart: the callee frame is built by
`handle_call_assembler` (`rpython/jit/backend/llsupport/rewrite.py:665-695`) and
the backend only loads `arglocs[0]` and calls the target.

This left an apparent CALLEE-side gap — "pyre does not yet run
`patch_new_loop_to_load_virtualizable_fields`, so a callee loop still carries
the vable-expanded inputarg list" — sourced from a comment in
`majit-backend-cranelift/src/compiler.rs` claiming the helper is held disabled
pending vable heap-writeback.  **That was false and is now measured.**  Both
sites were instrumented and the whole 313-benchmark corpus swept: the helper
truncated **2044 times with zero early returns** (`inputargs=14/15/16 → reds=2`),
and the caller-side resolver saw **38/38 CALL_ASSEMBLER targets at the expected
arity**.  The shrink has been universally active since `driver_descriptor()`
started returning `Some(...)`; the comment predated that flip.  The dead
fallback it guarded, and the `num_scalar_inputargs` plumbing that fed it, are
deleted.

**`_MULTIFRAME` — the ON path is the upstream structure.**  Upstream's
`convert_and_run_from_pyjitpl` (`rpython/jit/metainterp/blackhole.py:1799-1826`)
*is* "convert the whole metainterp framestack into a chain of
BlackholeInterpreters and run it", and it is what upstream does when a residual
call forces the virtualizable (`vable_after_residual_call` →
`SwitchToBlackhole(ABORT_ESCAPE)`, `pyjitpl.py:3389`).  pyre's port is
line-faithful.  The OFF path — decline to escape/replay — has no upstream
analogue at all; upstream cannot decline (`assert False  # ^^^ must raise`,
`pyjitpl.py:2956`).  Two honest qualifications: (a) upstream also hands control
back to the plain interpreter, via `ContinueRunningNormally` re-invoking the
portal (`blackhole.py:1067-1069`, `warmspot.py:970-982`), so "upstream never
returns to the interpreter" is *not* the reason OFF is a deviation — the reason
is the blanket decline and the rewind; (b) upstream needs no per-frame
virtualizable because every non-standard virtualizable degrades at trace time to
concrete `getfield_gc` / `setarrayitem_gc` on the real heap frame
(`_nonstandard_virtualizable`, `pyjitpl.py:1120-1146`), leaving all callee
PyFrames heap-authoritative before the chain is built, whereas pyre's walker
keeps inlined callee frames unmaterialized (`fbw_strict_fold_frame_reg`,
`vable_ops.rs:184-192`).  So the remaining work is a materialization step
upstream does not have — a consequence of pyre's virtual-callee-frame inlining —
and that is the whole of it.  The per-frame vable binding this section used to
list beside it is already done: `PyjitplBlackholeFrameConfig` carries a
`per_frame` slice and `convert_and_run_from_pyjitpl` overrides each level's
`virtualizable_ptr` / `virtualizable_stack_base` from it (`blackhole.rs`), so
every level already runs against its own frame instead of a shared pointer.

**Resolved 2026-07-26 — the adopt's root-mismatch decline.**
`try_adopt_multi_frame_blackhole` (`pyre-jit-trace/src/trace.rs`) declined
whenever the recovered chain's root was not the walked frame, and its comment
attributed that to "a chain rooted at an intermediate frame", naming the
`jit.virtual_ref` emit at the inline push as the prerequisite.  Both halves of
that attribution were wrong.  The chain *is* rooted at the walked frame; the two
sides of the comparison were two representations of it.  `per_frame[0]` is
recovered from the trace's frame register, whose root vable identity
`seed_virtualizable_boxes` bakes against the **live** frame address
(`set_live_vable_frame_addr`, set before `init_symbolic` precisely so it is not
the discarded snapshot's), while `cf_addr` is the **snapshot** copy.  Five
events printed `per_frame[0] == live == 0xa4be2db40` against
`cf_addr == 0xa4c0515e8`, so the decline was unconditional and no `VIRTUAL_REF`
emit was involved.  (The emit is still absent — `opimpl_virtual_ref` / `_finish`
are ported in both `majit-metainterp/src/pyjitpl.rs` and
`pyre-jit-trace/src/state.rs` and **neither has a caller outside a `#[test]`**,
so `virtualref_boxes` is empty in every live trace.  That is a real gap; it was
simply not this one.)

The fix points the two *identity* uses at the live address, under the same
`!= 0` fallback the identity bake itself uses, and leaves every other use where
it was:

| use of the walked frame | address | why |
|---|---|---|
| root-mismatch comparison | live | must be asked against the address the identity was baked from |
| root `f_backref` operand | live | the `ptr::eq` skip has to fire for `frames[0]`; the snapshot is freed at walk end, so linking to it would leave a dangling `f_back` for a later `sys._getframe().f_back` |
| `apply_blackhole_crn` | snapshot | the portal epilogue propagates snapshot → live, the same contract the single-frame arm relies on |
| `drive_multi_frame_blackhole` vable root + `stack_base` | dead | `per_frame` is always `Some` here and overrides both, per level |
| `concrete_nlocals`, the `ec` read | either | pycode-derived, and the snapshot copies `execution_context` verbatim |

Opening the comparison exposes one consequence that could not fire while it was
shut: frame 0's blackhole level runs against `per_frame[0]`, the **live** frame,
so its `setfield_vable` stores land there while `apply_blackhole_crn` writes the
snapshot — and the epilogue then copies the snapshot's *whole* locals array onto
the live frame (`restore_resume_state_from`), reverting every such store the CRN
write does not happen to cover.  The adopt therefore folds the live frame's
state into the snapshot before the CRN write, restoring "the snapshot is the
committed image".

**Measured 2026-07-25: the multi-frame path has no corpus coverage.**  The
vable-escape latch site was instrumented and all **318** benchmarks
(`pyre/bench` + `pyre/bench/synth`) were historically run with the multi-frame
gate enabled.  The site was reached in **3 benches**
(`getframe_escape_flush_writethrough_regression`,
`synth/getframe_inlined_callee_own_frame`, `synth/getframe_stored_fback_walk`),
5 events each, and **all 15 have `inline_subwalk=false`** — every one takes the
single-frame arm and adopts.  `build_multi_frame_miframe` is therefore never
called, the image is never latched, and the adopt never sees a candidate.  So
flipping `_MULTIFRAME` ON is a no-op across the corpus and none of the items
above is exercised.  Building a benchmark that reaches `inline_subwalk=true` at
a vable escape was the prerequisite, and **that benchmark now exists**: a
`while`-driven loop calling a straight-line inlined callee that calls a
zero-argument `sys._getframe` reaches the site.  `for` is what every existing
`getframe_*` bench gets wrong — with a FOR_ITER item in flight the callee's
nested residual is declined by `fbw_abort_nested_unjournaled_residual` before
`execute_residual_call` runs, so the force never happens inside the sub-walk.
Under that shape `build_multi_frame_miframe` **succeeds at depth 2**, so the
build side was never what blocked.  It is landed as
`synth/getframe_while_inlined_callee_subwalk`; the three shape choices in its
header are load-bearing and changing any of them silently stops exercising the
path.  With the comparison fixed, that fixture historically reported **5
`BUILT multi-frame depth=2` and 5 `adopted multi-frame terminal`,
zero declines** (the other 5 escapes in the run have `inline_subwalk=false` and
take the single-frame arm, as before), and prints the same result as CPython and
PyPy.

**What the build still declines, and why the decline is right.**  Two shapes
reach the latch and are then refused by `capture_inline_parent_blackhole`
(`resume_snapshot.rs`): a caller with an exception handler around the inlined
call, and two nested inlined levels (depth 3).  Instrumented 2026-07-26, both
report the same cause — a ref color that is **live at the caller's post-call
coordinate holds `ConcreteValue::Null`**:

```text
try/except caller: ref color=11 not concrete: Null  result_color=Some(5) nlocals=3 depth=3 live_ref=[0,1,2,5,11]
depth 3:           ref color=2  not concrete: Null  result_color=Some(0) nlocals=1 depth=1 live_ref=[0,1,2]
```

Neither is the not-yet-produced result slot, and neither involves the bridge
parent-frame constructors — every `[s2-gate]` event in both runs prints
`not_bridge=true`, and the latch requires `!is_bridge_trace`, so those
constructors are unreachable from here.  `ConcreteValue::Null` is the
**untracked** sentinel, deliberately distinct from `Ref(PY_NULL)` = "uninitialised
local" (`state.rs`, `trace_opcode.rs`), so accepting it would fabricate a parent
frame rather than reproduce one.  Declining is correct; closing these two shapes
is the outer-locals materialization named above — completing the caller's
concrete banks at an inline escape — not a change to the capture itself.  Both
are pinned by `synth/getframe_while_subwalk_decline_shapes` so a decline cannot
silently become a wrong answer.

**The flip was blocked by a wrong answer, not a decline — resolved 2026-07-30.**
Measured 2026-07-26.  The walker executes residuals **concretely** while an
inline push did not run the interpreter's call sequence, so `ec.topframeref`
still named the CALLER while an inlined callee body ran.  A `sys._getframe`
that is *itself* the escaping residual therefore read the wrong frame at walk
time, and the adopt committed that answer where legacy escape/replay discards it:

```text
_gf().f_code.co_name   -> "main",     not "leaf"
_gf(1).f_code.co_name  -> "<module>", not "main"     # one level too far up
_gf(1).f_locals["k"]   -> KeyError                   # same cause, seen through the argument
```

**One wrong iteration per multi-frame adopt** — 5 adopts, 5 wrong, in each part
of `synth/getframe_while_escaping_read_frame_identity`, the acceptance test.
This was *not* outer-locals staleness.  A `sys._getframe` executed **after** the
escape, inside the blackhole, is correct — the chain publishes each level's
frame as it runs — and an in-blackhole read of a caller local mutated earlier in
the same iteration was measured correct against CPython and PyPy.  Closing it
needed the inlined-call push to publish the callee frame on the execution
context, which is what `walker_ec_enter` / `walker_ec_leave` do; the
`jit.virtual_ref` emit rode along with them.  So the original decline comment was
right that an inline-push `enter` is the prerequisite, and wrong only about which
check it gated.

**Identity half resolved 2026-07-30.**  With that push landed, the same fixture
reports `30000 30000 0 0` — zero wrong frames — under CPython and under pyre, and
the chain-root identity gate no longer declines these shapes.  The gate was
flipped default-ON and then retired outright (§3).

**But the flip was not sound.**  A SECOND blocker was recorded in
`try_adopt_multi_frame_blackhole`'s own comment and missed when the flip was
verified: only frame 0's locals were published, so an inlined callee's frame array
kept its pre-sub-walk contents while every LOAD_FAST reads that array.  With the
flip live, an inlined callee that stores `e.__traceback__` and then reads an
attribute off it resumed the local as null and faulted in `object_getattr_miss`
— a hard SIGSEGV, reproduced at `0f9c371b63`.  Two PR reviewers flagged it
independently, and `synth/blackhole_inlined_callee_local_after_escape` pins the
crashing shape.  It is closed by mirroring an inlined MIFrame's standard-vable
writes onto that level's own concrete red frame at the time they are made
(`current_inline_concrete_frame`, `store_live_frame_static_int`), so a level is
resumable from the frame it already owns rather than from a publication step
that runs at the adopt — one red frame per MIFrame, which is the shape the
codewriter's `getarrayitem_vable_r` lowering assumes.

**A THIRD blocker sits under the second.**  With the levels resumable and no
crash, the fixture still returned a silently wrong `[(False, 2), (True, 2)]`
against `[(True, 2)]`, at exactly one wrong iteration per adopt.  At every
mismatch the escaping `sys._getframe()` returned the chain's level-1 frame
(`per_frame[1]`, the frame the seed built and the sub-walk runs on) while the
traceback named a *different* frame object for the same invocation.  The producer
is `record_inline_traceback_for_recording`: the walk-time concrete traceback node
for an inlined level was anchored on a frame the hook `createframe_obj`s from the
promoted code and globals.  That hook predates the seed, and the level now has a
real frame — the same object the EMITTED node already names, since
`traceback_node_site` resolves its frame operand from the level's frame register.
So the walk and the compiled run disagreed, and only the walk's answer is
committed by an adopt, which is why the corpus saw it exactly once per adopt and
never in steady state.  `record_inline_application_traceback` now anchors the
concrete node on that frame and falls back to the fabricating hook only for a
level inlined without one.

Anchoring on the real frame moves one obligation along with the node.  The
fabricated frame carried the raise coordinate because the hook stamped it, while
the level's own frame carries the entry sentinel: the recording walk does not
make `dispatch_bytecode`'s per-opcode `last_instr` store, and a frame that leaves
by the exception never reaches an exit that would publish one either, so
`f_lineno` answers the `def` line.  `synth/exception_traceback_frame_lineno`
reads exactly that, as a second `('raises_out', 1, 0)` shape beside the correct
`('raises_out', 1, 1)`.  The anchor therefore makes the same store the blackhole
already makes for its replay in `publish_last_instr_at_live_marker`.

**All three are closed, and the arm adopts.**
`synth/getframe_inline_subwalk_multiframe` measures 5 builds / 5 adopts / 0
declines, `..._while_escaping_read_frame_identity` 10 / 10 / 0 and
`..._while_inlined_callee_subwalk` 5 / 5 / 0, all with unchanged output, and
`synth/blackhole_inlined_callee_local_after_escape` matches the reference.

**The RUNTIME half of that anchor was investigated and declined.**  Only the
walk-time record was moved onto the level's own frame; the `emit_runtime` arm of
`record_inline_application_traceback` still emits the frame-fabricating hook.
Two things came out of measuring it, and both are worth keeping:

*It is unreachable.*  An lldb breakpoint on that arm's own call-descr
construction counts zero hits across the 43 corpus exception fixtures and 15
hand-built probes, corroborated by a `MAJIT_LOG` scan finding no call with the
hook's `[Ref, Ref, Ref, Int, Int]` signature in any dumped trace.  The mechanism
is that `record_prepend_application_traceback` never declines: `emit_runtime` is
its negation, and the `exc.is_constant()` arm it would decline on is suppressed
because every raising residual assigns `class_of_last_exc_is_const = false`
immediately before `walker_record_guard_exception` reads it.  So the fabricating
hook reaches no compiled traceback today — it is still called, but only from the
walk's own no-frame fallback inside a bridge sub-walk.

*Porting it would break a documented allocation contract.*  The obvious port —
emit the pointer-taking hook with the level's frame operand, the shape the
top-level sibling already uses — cannot be applied here.  Every frame that
reaches `record_application_traceback` today is a non-moving oldgen block, which
is exactly what `w_pytraceback_new` relies on when it roots `w_next` and `w_code`
but deliberately not `frame`.  The top-level sibling passes the standard
virtualizable, the walk passes a `FrameBox`, and the fabricating hook passes its
own `createframe_obj` frame — all oldgen.  A compiled trace's inlined callee
frame is not: it is the trace's own `NewWithVtable`, which the GC rewriter lowers
to a nursery allocation.  Handing that to the recorder would hold a movable
pointer across the parking allocation inside `w_pytraceback_new` and store a
pre-move address into `PyTraceback.frame`.  The port therefore needs the
root-and-reload shape on the recorder first, which also covers the same
pre-existing exposure on its `w_next` argument.

**That contract is a pyre deviation, and it is not the traceback's.**  Upstream
attaches no allocation rule to a traceback's frame at all: `pytraceback.py:29`
stores an ordinary traced field of an ordinary movable `pyframe.py:52 class
PyFrame(W_Root)`, nothing under `pypy/interpreter/` calls `rgc.pin` (whose own
doc, `rpython/rlib/rgc.py:88-97`, rules out the lifetime use), and a minor
collection relocates the frame and rewrites every referring slot
(`rpython/memory/gc/incminimark.py:2237` / `:2252`) because roots arrive as slot
addresses (`rpython/memory/gctransform/shadowstack.py:43-46`) and compiled code
re-reads the frame after each collecting call
(`rpython/jit/backend/x86/assembler.py:1369-1377`).  The one obligation upstream
does attach is a JIT one — force the vref, `error.py:370
tb.frame.mark_as_escaped()` — which pyre already has.

The traceback edge itself is likewise already upstream-shaped:
`pytraceback_object_custom_trace` forwards `frame` as a *mutable* slot, so a
relocation would be written back today.  What forbids a movable frame is
elsewhere — raw `*mut PyFrame` duplicates no root walker reaches.  `FrameBox`
holds a forwarding-capable `owner_root` and never reads it back, so every
`Deref` goes through the stale raw field; `eval_loop` runs behind a
`&mut PyFrame` across a safepoint (the exact class RPython's translated
shadowstack enumerates for free); the blackhole keeps the virtualizable as a
bare integer, as did `INLINE_CONCRETE_FRAME`.

The first two of those are now closed: `FrameBox` reads its frame back out of
its own `owner_root` instead of the raw field it had cached (and `into_raw`
reads the slot before releasing it, which it previously did in the opposite
order), and `INLINE_CONCRETE_FRAME` is held as a root rather than a bare
`Cell<*mut PyFrame>`.

**There is no "virtualizable reload" to port** — an earlier revision of this
section named one, and it does not exist upstream.  `_reload_frame_if_necessary`
reloads the JITFRAME alone (`ebp` from the root-stack top,
`rpython/jit/backend/x86/assembler.py:1369-1377`), which pyre already has in the
dynasm aarch64 assembler; `rg reload_frame_if_necessary rpython/jit/backend/`
turns up no vable counterpart on any target.  Upstream needs none: the
virtualizable is an ordinary GCREF box (`metainterp.virtualizable_boxes[-1]`),
so it rides the same gcmap-covered jitframe slots as every other ref across a
collecting call, and `llsupport/assembler.py:301` notes it is deliberately "not
in a register".

What is left is the remaining raw copies — and chasing them to the end shows the
non-moving frame allocation is not a deferrable TODO but a forced adaptation.

`eval_loop` holds `frame: &mut PyFrame` across the loop and hands it to
`execute_opcode_step`, which allocates.  The JIT eval loop already runs the
RPython discipline for its own copy — `FrameRoot` pushes the frame on the shadow
stack, caches the `ShadowStackSlot`, and every collection point is followed by a
fresh `let f: *mut PyFrame = frame_root.frame()`, with the comments naming
`execute_opcode_step`, `handle_exception` and the ec block as the points.  It is
load-bearing there because a JIT-inlined callee frame really can be nursery-
resident (`emit_new_pyframe_inline_with_params` lowers to a nursery bump).

But that discipline protects only the LOOP's copy.  The reference handed *into*
`execute_opcode_step` is live across every allocation the opcode performs, and
so is every frame reference below it.  Making frames movable therefore requires
re-deriving the frame after each collection point inside every handler — which
is precisely what RPython gets for free, because the translator rewrites live
GCREF locals in their shadow-stack slots
(`rpython/memory/gctransform/shadowstack.py:43-46`) and the interpreter never
writes a reload by hand.  Rust has no such pass, and hand-writing it across the
opcode implementations is neither reviewable nor maintainable.

So the enabling condition for movable frames is a mechanism, not a patch: some
automatic rewrite of live frame references across collection points.  Until one
exists, `FrameBox::new` allocating non-moving is the thing standing in for it,
and it should be read as pyre's stand-in for the translator's shadow-stack
rewrite rather than as an accident awaiting cleanup.  The same holds for what it
props up: `PyTraceback.w_code` stays, and so does the conditional frame edge —
though the edge's reachable case has narrowed to the pre-hook bootstrap frame
(see `TraceCtx::virtualizable_heap_ptr` below), which is why the guard is now
kept for its failure mode rather than for a case anyone has observed.

Two cached raw copies were checked against upstream individually rather than
left on a list.  Neither is a hazard, and only one is even the deviation it
looked like:

*The blackhole's `virtualizable_ptr: i64` — not the field it resembles.*
Upstream's `BlackholeInterpreter` has no such field: every vable bhimpl takes
the virtualizable as an explicit argument (`blackhole.py:1374`
`bhimpl_getarrayitem_vable_i(cpu, vable, index, fielddescr, arraydescr)`, and
the same shape for the `set*`/`arraylen` family), sourced from the register
bank, which pyre roots too (`push_bh_regs`).  Pyre's field is not that carrier.
It is the frame identity pyre's blackhole traceback recording needs —
`record_frame_traceback` passes it to `record_application_traceback_for_recording`
— and upstream has no counterpart to that at all, because its blackhole does not
record application tracebacks.  Lifetime is already covered: the frame is on the
interpreter chain, and `run` additionally pushes the vable's array-field slots
onto the resume-ref root stack for the whole run
(`VirtualizableInfo::push_resume_ref_roots`, which forwards those slots in
place).  What the field does not do is forward *itself*, and that is exactly
what the non-moving contract above makes unnecessary.  Rooting it would forward
nothing.

*`TraceCtx::virtualizable_heap_ptr` — half closed; the rest is architectural.*
It caches what upstream unwraps per use (`pyjitpl.py:3472-3474
synchronize_virtualizable` re-derives the write target from
`virtualizable_boxes[-1]` every time), and a root portal seed points it at the
`snapshot_for_tracing` copy while baking the identity against the live frame,
so the two name different objects where upstream has one.

The half that was a rooting problem is gone.  The snapshot used to be a
`FrameBox::new_boxed` allocation the GC did not own, which is what left the
cached pointer with no slot to forward — and, downstream, what made the
traceback the walk records against it (`pyjitpl.rs`
`record_application_traceback(excvalue, self.vable_ptr, frame)`) carry a frame
pointer that dangled as soon as the walk ended.  `FrameBox::new`'s `owner_root`
now supplies the root that was missing, so `snapshot_for_tracing` allocates
GC-owned, the traceback's conditional frame edge forwards and keeps it alive,
and a full-corpus probe on that edge observed no non-GC frame reaching a
traceback at all.

The half that remains is not a rooting change and not a tracer change either.
The two objects exist because pyre's tracer steps a *copy*: the walk executes
the iteration concretely against the snapshot while the live frame stays parked
at the loop header for the compiled loop to run from.  That forces the split in
both directions — the identity must be the live frame or
`patch_new_loop_to_load_virtualizable_fields` gets a frame unrelated to the
boxes (`compile.py:458 assert i == len(inputargs)`), and the synchronization
target must be the snapshot or `refresh_virtualizable_shadow_from_heap` reads a
frame the walk never mutated and the shadow drifts.  Upstream needs one field
because its metainterp *is* the interpreter for that iteration.  So this cache
converges when the copy-walk does, and not before.

**The invariant is narrower than "frames are non-moving", and that is by
design.**  A compiled trace's own `NewWithVtable(pyframe_size_descr())` really
does lower to a nursery bump, so JIT-emitted frames are movable — the audit left
this inferred, and it is now measured.  Making it uniform is one line, with a
direct in-tree precedent: the `W_ObjectObject` group marks its size descr
`non_moving` for the identical reason ("the instance layer reaches an instance
through raw pointers it does not root"), and `rewrite.rs` then declines the
nursery and lands on the old-gen `gen_malloc_fixedsize`.  Marking the PyFrame
group the same way costs:

| bench | nursery frames | non-moving frames |
|---|---|---|
| `fib_recursive` | 1193 ms | **9249 ms (7.75x)** |
| `inline_helper` | 219 ms | 213 ms (0.97x) |
| `nested_loop` | 298 ms | 302 ms (1.01x) |

PERF-CLAIM-UNVERIFIED: this table.  The command that produced it, the sha of
each arm, and the round count were never recorded, so it cannot be reproduced
or falsified.  `check.py` did not produce it — `check.py:174` returns
`("", 0.0, 124, "")` on `TimeoutExpired`, so the harness cannot report 9249 ms
for a run it killed, and the 1193 ms denominator is stale (`fib_recursive` runs
in ~650 ms today).  What *is* doubly attested is the `check.py` result of
applying the change: `FAIL dynasm fib_recursive timeout (>5s)`, 1 failed / 353
passed, with no wrong output anywhere.  Read the effect as **">4x, direction
certain"** and the table as an unsourced elaboration of it.  The conclusion
below does not depend on the precision.

**Scope, which is the sharper limit on this experiment.** The flag it toggled
has exactly one production reader — `rewrite.rs:1091`, on the JIT's
`NewWithVtable` lowering.  `FrameBox::new` never consults a descr: it calls
`try_gc_alloc_stable_raw` unconditionally (`pyre-interpreter/src/pyframe.rs`)
and lands on `alloc_oldgen_typed` (`pyre-jit/src/eval.rs`).  Interpreter frames
were therefore old-gen in **both** arms, and the measurement says nothing about
them.  Every conclusion here is scoped to JIT-emitted frames.

Also worth stating because it bounds how far this can be generalized: upstream
cannot express the arm that was measured.  `rpython/jit/codewriter/jtransform.py:1012-1015`
rejects the flag outright — `if d.get('nonmovable', False): raise UnsupportedMallocFlags(d)`
— and upstream's own `malloc_big_fixedsize` still takes the nursery for
frame-sized objects (`jit/backend/llsupport/rewrite.py:778-788`).  The penalty
is the cost of enabling a behaviour with no upstream counterpart, not the cost
of an upstream design pyre declined.

So, within that scope: the nursery frame path is not merely reachable but hot on
recursive calls, and making JIT-emitted frames uniform with interpreter frames
is unaffordable.

What the system actually maintains is the narrower rule: *a frame that escapes
into raw-pointer-holding territory is old-gen; a frame that stays inside a
compiled trace may be nursery.*  The two seam-local props below are how that is
enforced — they are the design, not an unmaintained accident, and the audit
reading them as belt-and-braces over an unreachable case was wrong.

An attempt to check the rule with an assertion turned up a second reason it
stays comment-enforced.  A `debug_assert!` in `w_pytraceback_new` that the frame is not
nursery-resident cost three `getattr_*` perf gates (`9.1x > 6x`, `8.9x > 5x`,
`8.1x > 5x`) even in a release build: `pyre-interpreter` is extracted to LLBC,
so the assertion is in the JIT's view of the function regardless of the host
profile, and `gc_is_nursery_object` became a real call on every traceback the
traced code builds — `rg gc_is_nursery_object build/llbc/pyre-interpreter.ullbc`
confirms it landed there.  A `debug_assert!` is not free in an LLBC-extracted
crate on a traced path.

The two props that carry the narrow rule are therefore load-bearing: the dynasm
runner routes a resume-materialized virtual `PyFrame` to oldgen on purpose, and
the JIT's traceback recorder builds a fresh oldgen frame instead of handing
`record_application_traceback` a materialized virtual.  Both sit exactly where a
nursery frame would otherwise cross into raw-pointer territory, and the
measurement above is why the crossing is guarded there rather than removed at
the allocator.  This is also why the runtime half of the inlined-level traceback
anchor stays declined: it would hand the recorder the one frame class the props
exist to keep away from it.

One thing the ON path already fixes: with a side-effecting inlined callee under
a `while` loop that returns from inside the loop, the OFF path runs the callee's
side effect ~5.2k extra times (the recorded trace-abort double-run class) while
the adopt gives the exact count.

Everything else that was thought to block the flip was measured and did not —
except the two gaps above, which the sweeps below did not cover because no corpus
fixture assigned a local in an inlined callee and read it back after the escape,
the one shape that reaches both: the
full corpus was **336/336 with the gate on (dynasm) and 336/336 with it off
(cranelift)** at the time, the blast radius is exactly `inline_subwalk = true` at
a vable escape (the latch is an `if`/`else if` whose single-frame arm requires
`!inline_subwalk`, so the multi-frame arm is the only one that shape can take),
and the two build-side declines above are correct.  Re-measured at the flip:
**1 failed / 341 passed on dynasm and on cranelift with the gate forced on**, the
one failure being a cpython/pypy reference mismatch unrelated to this path.

The multi-frame latch shares the outer `writes_live_heap`, odometer-unchanged,
non-bridge, blackhole-result, and resolvable-snapshot conditions with the
single-frame latch, then takes the `inline_subwalk` arm when its frame-stack
image builds successfully. The pre-existing `[s2-gate]` eprintln (under
`PYRE_FBW_DEBUG_ABORT`) already reports `inline_subwalk` at that site.

**A second coverage benchmark, 2026-07-27.**
`synth/getframe_inline_subwalk_multiframe` (#798) reaches the latch with
`inline_subwalk=true` and drives `build_multi_frame_miframe`. In the historical
gate-enabled measurement with `PYRE_FBW_DEBUG_ABORT=1`, it printed 5 `[s2-gate]
inline_subwalk` lines each followed by `[s2-build-decline] BUILT multi-frame
depth=2`.  One `sys._getframe(1)` level does not get there; the chain needs a
residual level under the walked frame and an inlined level under that, so the
force has to reach two frames up.  Per-frame vable binding, outer-locals
materialization, and the `jit.virtual_ref` emit are therefore validatable now.

## §2 — Not gates (12): Rust identifiers, not env vars

The audit regex matched non-env identifiers. These are real code; **do not
delete, do not count as gates.**

- `PYRE_STR_DESCR`, `PYRE_STR_BYTE_LEN_DESCR`, `PYRE_UNICODE_DESCR`,
  `PYRE_UNICODE_LEN_DESCR` — field-descriptor `const`s (`pyre-jit-trace/src/pyre_cpu.rs`)
- `PYRE_CLASS_DESCRIPTOR` — macro-built identifier `W_{}_PYRE_CLASS_DESCRIPTOR` (`pyre-macros`)
- `PYRE_CLASS_DESCRIPTORS` — the whole-program `linkme` distributed slice the
  macro registers into (`pyre-object/src/lltype.rs`); the only `PYRE_*` name that
  appears outside Rust and Python source, and it is not an env var
- `PYRE_PARAM_NAMES`, `PYRE_PARAM_REQUIRED` — macro `const __PYRE_PARAM_*` (`pyre-macros`)
- `PYRE_JIT_GRAPH_MODULES` — compile-time `const &[&str]` module manifest (`generated.rs`)
- `PYRE_REF_OPAQUE` — `OpaqueType::gc("PYRE_REF_OPAQUE")` type label (`annotator/builtin.rs`)
- `PYRE_JIT_DISABLED` — a `OnceLock<bool>` cache name holding the `PYRE_JIT==0` result (`pyre-jit/src/eval.rs`); the env var is `PYRE_JIT`
- `PYRE_STACKTOOBIG` — `pub static PyreStackTooBig` runtime symbol (`stack_check.rs`)

## §3 — Dead (13): no env read site

No source reads these. Comment-only or absent. **Historical measurement notes
are preserved in place per N7** (they record why code was deemed dead / what a
census verified); they are not live gates and cost nothing.

| var | state |
|---|---|
| PYRE_50 | absent — zero occurrences |
| PYRE_RTYPER | comments/diag-labels only; the real/legacy dual-gate runs unconditionally |
| PYRE_STRUCT_DIFF | comment only (`front/mir.rs`) — reference removed 2026-07-05 |
| PYRE_REQUIRE_MIR_FRONTEND | module-doc mention only (`front/mod.rs`); the doc claimed check.py sets it, but the LLBC requirement is unconditional — stale claim removed 2026-07-05 |
| PYRE_VSTACK_USE | planned flag, never wired (`jitcode_dispatch.rs` design notes) — vaporware references removed 2026-07-05 |
| PYRE_PATH3_VERIFY_STACK_READ | retired probe; "zero mismatch" note kept |
| PYRE_REMAP_PROBE | retired probe; "0 fires 2026-06-11" measurement kept |
| PYRE_S8B_HARNESS | retired census; "82/82 agreement" measurement kept |
| PYRE_MODULE_LOOP_TRACE | retired switch; historical note kept |
| PYRE_FULL_BODY_WALK | retired switch; the full-body walk is the sole tracer, so the OFF path (the deleted trait leg) is gone (#344) |
| `_MULTIFRAME` | retired switch; reader and OFF path deleted once `walker_ec_enter` / `walker_ec_leave` closed the escaping-`sys._getframe` identity answer (§1d). Flipped default-ON and retired 2026-07-30; `_MULTIFRAME_DEPTH` is a separate live depth bound and is not this gate |
| `_BLACKHOLE_RESUME` | retired switch; reader and OFF path deleted after #754 closed, with the multi-frame twin's retirement unblocking removal; it was flipped default-ON on 2026-07-25 |
| `PYRE_CARRIER_EXC_RESUME` | retired experiment; reader (`carrier_exc_resume_enabled`), the `setup_bridge_sym` pre-seed it guarded, `TraceCtx::bridge_guard_exc` and the `guard_exc` parameter of `start_bridge_tracing` all deleted 2026-08-06. The ON path measured inert — structurally redundant with the ungated walk-start `seed_standing_exception_for_walk` on the single-frame leg, and never exercised on the multi-frame carrier leg it was written for. §1b keeps the seed-site probe and both corpus runs |

## §4 — Live default-ON gates KEPT (retire when the epic closes)

Each is default-ON but still a load-bearing kill switch for an open rework; its
OFF path is a needed safety net. Retire at the listed trigger (A7).

| var | subsystem | retire when |
|---|---|---|
| PYRE_TWO_PHASE_RTYPE, PYRE_TUPLE_PER_SHAPE_CLASSDEF | rtyper prepass / per-shape tuple classdef | WS2 / #346 rtyper epic |
| PYRE_ORIGINAL_BOXES | greens++reds original_boxes index shape | box-identity #202 / resume F1 |
| PYRE_MIR_FRAMESTATE | framestate-threaded MIR lowering | MIR front-end #176/#181/#346 |
| PYRE_GC_ITEMSBLOCK, PYRE_GC_PREBUILT_REMEMBER, PYRE_GC_INTERP, PYRE_GC_INTERP_COLLECT | GC-managed items / prebuilt minor-skip / interpreter allocation + collect rollback | WS3 / #355 / F3 GC rework |
| PYRE_CL_NO_CLOSING_JUMP | cranelift attached-loop closing jump | #245 cranelift perf (explicit rollback hatch) |

`PYRE_GC_INTERP` is default-ON on every target. Its OFF path still selects the
unmanaged `malloc_typed` stepping-stone allocation and remains a rollback hatch
until translated shadow-stack roots make the ordinary moving-nursery path safe.

## §5 — Other live gates (not removal targets by the "already-ON" criterion)

Kept as-is; listed for completeness.

- **Diagnostics (~34, default-OFF)** — print/log/dump/probe/assert only; tooling,
  not experiments: `PYRE_FBW_DEBUG_ABORT`, `_INLINE_DIAG`, `_MF_DIAG`,
  `_STRICT_DIAG`, `PYRE_WALK_PERFN_JITCODE`, `PYRE_DUMP_PERFN_JITCODE`,
  `PYRE_P2_DIAG`, `PYRE_PCDEP_VALIDATE`, `PYRE_MIR_FRAMESTATE_DEBUG`,
  `_FRAMESTATE_STRICT`, `PYRE_MIR_FRONTEND_DEBUG`, `PYRE_VSTACK_DIAG`,
  `PYRE_PROBE_AUTHORITATIVE`, `_BH_STARTUP`, `_SNAPSHOT`, `_SUBSCR`,
  `PYRE_S9_PROBE`, `PYRE_PROFILE_DRAIN`, `_PIPELINE`, `PYRE_MFRAME_DIAG`,
  `PYRE_RTYPER_VERBOSE`, `PYRE_JTRANSFORM_SHADOW`, `PYRE_DIAG124C`, `_51C`,
  `_GIN`, `_INLINE_RECOG`, `PYRE_WASM_DUMP_ALL_TRACES`, `_DUMP_BAD_TRACE`,
  `_EXEC_TRACE`, `_JIT_STATS`, `PYRE_INTERP_RETURN_LOG`, `PYRE_NBODY_DEBUG`,
  `PYRE_DEBUG_CALL`, `PYRE_DEBUG_CLASS`.
- **Default-OFF experiments (1 remaining)** — triaged in §1b/§1c (4 retired
  in the 2026-07-05 pass, 9 retired since then; `PYRE_P2_DRAIN` retired with
  the framestack-walk deletion; `_VABLE_SCALAR_CA` retired 2026-07-25, see
  §1d; `PYRE_CARRIER_EXC_RESUME` retired 2026-08-06 with its ON path deleted,
  see §1b).  Kept: `_CALLEE_VSTACK` (callee-local operand-stack mirror).  Its
  *ON* path is the unattested one, so it is an adoption target rather than a
  retirement target.
  The single-frame resume-past-escape switch graduated out of this bucket on
  2026-07-25 when it flipped default-ON. It is now retired alongside the
  multi-frame switch, with both readers and OFF paths deleted after
  `walker_ec_enter` / `walker_ec_leave` closed the escaping-`sys._getframe`
  identity answer.

  `_CALLEE_VSTACK` was evaluated for a flip on 2026-07-25 and **declined —
  the ON path is a half-finished port with no consumer**.  Parity first:
  upstream has *no* per-frame operand-stack model to port, because the
  codewriter has already flattened the Python stack into jitcode registers —
  `MIFrame` carries only `registers_i/r/f` (`pyjitpl.py:65-95`), a callee's
  resume section is its own registers under the per-pc liveness window
  (`get_list_of_active_boxes`, `pyjitpl.py:177-234`), and `virtualizable_boxes`
  is established once for the ONE standard vable
  (`initialize_virtualizable`, `pyjitpl.py:3314-3331`; `_nonstandard_virtualizable`,
  `1120-1146`).  So NEITHER side of this gate is an upstream port, and the
  mirror's own module doc concedes it has no `rpython/jit/metainterp/`
  counterpart.  What decides it is a defect on the ON side: with the gate on,
  `seed_callee_vstack_mirror` fills the mirror with CALLEE content
  (`vstack_mirror.rs:727-742`), but the two maintenance sites still classify
  `index_value` against the OUTER portal sym's `nlocals()`
  (`vable_ops.rs:558-570`, `610-645`), and `collect_call_stack_overrides`
  indexes the mirror with `caller_sym.nlocals()`
  (`resume_snapshot.rs:1138-1151`) — callee content read through caller frame
  geometry.  Meanwhile the guard-time consumer is unreachable in a sub-walk
  (`resume_snapshot.rs:325` gates it behind `!inline_subwalk`), which is why a
  306-bench corpus A/B on both backends changes the `[vstack-reconcile]` count
  in 60 benches with byte-identical output and no measurable timing difference:
  the mirror is seeded and then read by nobody.  Definition of done before
  re-evaluating: make the maintenance sites read the ACTIVE callee jitcode
  metadata (what the gate doc already asks for), then land a consumer.
- **Config / value / master switches (~17)** — tuning, paths, modes; keep:
  `PYRE_WALKER_STORE_SUBSCR_FNADDR`,
  `PYRE_MIR_FRONTEND_LLBC`, `PYRE_WASM_ENGINE`, `_FUEL`, `_MODULE`, `_NO_CACHE`,
  `PYRE_GC_INTERP`, `PYRE_JIT`, `PYRE_NO_JIT`, `PYRE_STDLIB`,
  `PYRE_CHECK_PYPY3`, `PYRE_CHECK_PYTHON3`, `PYRE_SANDBOX_NO_SECCOMP`,
  `PYRE_SHARED_BUILD`, `PYRE_SYNTH_PYPY`, `PYRE_SYNTH_PYRE`, `PYRE_SYNTH_PYTHON`.
- **Test harness (1)**: `PYRE_MIR_STRESS_LLBC`.

## §6 — The 66 gates the audits never listed (2026-08-07)

The hand audits above enumerated what they were looking at. Measured against the
tree instead, **66 of the 105 live gates had no entry anywhere in this file** —
the table was 63% empty, because nothing failed when a new gate skipped it.
`pyre/pyrex/tests/gate_triage_complete.rs` is now that failure: a `PYRE_*` read
with no entry here fails `cargo test`. The counts to quote, distinguished:

| count | value |
|---|---|
| distinct names read from the environment | **111** |
| — of those, read from Rust | 105 |
| — read only from the harness Python | 6 |
| (file, name) read pairs | 137 |
| **live gates that were absent from this file** | **66** |
| names still listed live with no read site left (retire) | 0 |

```sh
{ git ls-files '*.rs'; git ls-files 'pyre/**/*.py' 'scripts/*.py'; } \
  | xargs rg --no-filename -o \
      '(env::var[_a-z]*|host_os::var|getenv|environ\.get)\(b?"(PYRE_[A-Z0-9_]+)"' \
      -r '$2' | sort -u
```

`--no-filename` is what makes this count gates: without it rg prefixes each hit
and `sort -u` counts (file, name) pairs instead. Each read form is here because
something was hiding behind it:

- `host_os::var` and `host_seam::ops::getenv` (a **byte** string) are how
  `importing.rs` reads `PYRE_STDLIB`. Neither adds a name — that gate is also
  read through `env::var` in `pyre-wasm-runner` — but a sandbox- or wasm-only
  gate would have had no such cover.
- `environ.get` and `getenv` are the harness. Six gates are read from
  `check.py`, `check_synthetic.py`, the `extra_tests` runners and
  `scripts/llbc_extract.py` and from no Rust file at all, so every `*.rs`
  census — including this section's first draft — missed all six.

Only unambiguous reads count. The harness also *writes* into a child's
environment (`env[…] = …`, `env.pop(…)`), and writing a gate for a child is not
owning it: the child's read is what this file is about. A subscript cannot be
told from a read without parsing, and naming a fixture variable here would enter
it in the census as a documented gate — which is why the example above has no
name in it.

**Spell every name in full at least once.** This file abbreviates runs of related
gates (`PYRE_WASM_ENGINE`, `_FUEL`, `_MODULE`), and the brake matches whole
tokens, so a name appearing *only* in that shorthand reads as undocumented.
`PYRE_SYNTH_PYRE` and `PYRE_SYNTH_PYTHON` were written `_PYRE`, `_PYTHON`, and
were the only two the widened census reported missing — they had been documented
all along. The shorthand is fine beside a full spelling; it is not fine alone.

**A retirement row documents nothing, wherever it sits.** §1/§1b/§1c/§2/§3 are
history sections and no name in them counts. But §1d's heading reads *Parity
verdicts*, so that section reads live while its table marks
`PYRE_FBW_VABLE_SCALAR_CA` **RETIRED** — a mixed section, which section
granularity cannot express. So any row that says "retired" is skipped too, and
re-introducing a reader for a retired gate fails the brake rather than passing on
the strength of its own obituary.

Polarity below follows this file's rule, with one correction it needed: an
`is_none()` whose value *is* the enable flag means default **ON**, but an
`if …is_none() { return; }` early-return guard means the thing is default
**OFF**. Three diagnostics (`PYRE_DESCR_SPELLING_GATE`, `PYRE_GC_DIAG`,
`PYRE_MC_DIAG`) read as ON under the unqualified rule and are OFF in fact.

### §6a — Live default-ON (4): the removal targets

| gate | what is ON by default | retire when |
|---|---|---|
| PYRE_JD1 | the jd1 compiled-loop experiment (`eval.rs jd1_experiment_enabled`); `PYRE_NO_JD1` or `PYRE_JD1=0` turns it off, and no-JIT implies off | the jd1 experiment concludes |
| PYRE_JD1_NO_ENTER | entering the compiled jd1 loop directly rather than leaving the drain to the interpreter caller | with `PYRE_JD1` |
| PYRE_WALKABORT_OFF | the non-carrier walk-abort leg (`trace.rs walk_abort_leg_enabled`) | kept deliberately: the leg commits irrevocably once the blackhole runs, so it is the one-binary A/B for the bug class it sits in |
| PYRE_WASM_FULL_TEARDOWN | skipping the ~0.2s wasm engine teardown at exit; setting it restores the drops for leak diagnostics | when teardown stops being the dominant fixed startup tax |

### §6b — VALUE knobs (11): config, not gates

`PYRE_DYN_INDIRECT`, `PYRE_FBW_MULTIFRAME_DEPTH`, `PYRE_JD1_THRESHOLD`,
`PYRE_OPTION_RESIDUAL_NARROW`, `PYRE_PCMAP_RECIPE_RESULTCOLOR_AUDIT_PROBE`,
`PYRE_TRACE_CALL_DIAG`, `PYRE_TRACE_OPS_DIAG`,
`PYRE_WASM_FORCE_CA_TERMINAL_DECLINE`, `PYRE_WASM_FUEL`,
`PYRE_WASM_GUEST_PROFILE`, `PYRE_WASM_MODULE`.

### §6c — Default-OFF diagnostics, censuses and probes (52): keep, cost nothing

Each is inert unless set, so none is a removal target by this file's
already-ON criterion. They are listed so they cannot be missed again.

`PYRE_BH_NULL_ARG`, `PYRE_CALLEE_RCA`, `PYRE_CATCH_LIVE_CENSUS`,
`PYRE_DESCR_SPELLING_GATE`,
`PYRE_DIAG_51C`, `PYRE_DIAG_GIN`, `PYRE_DIAG_INLINE_RECOG`,
`PYRE_DYNASM_EXEC_DIAG`, `PYRE_FBW_CENSUS`, `PYRE_FBW_INLINE_DIAG`,
`PYRE_FBW_LOOPBODY_SCAN_FULL`, `PYRE_FBW_LOOPBODY_SCAN_LOOP_ONLY`,
`PYRE_FBW_MF_DIAG`, `PYRE_FBW_STRICT_DIAG`, `PYRE_FIELD_IDENTITY_CENSUS`,
`PYRE_FORITER_INFLIGHT_CENSUS`, `PYRE_FOR_ITER_GATE_DIAG`,
`PYRE_GC_DIAG`, `PYRE_GC_FREELIST_DIAG`, `PYRE_JD1_DEBUG`, `PYRE_JD1_DUMP`,
`PYRE_LB_SITE`, `PYRE_LLBC_SKIP_FINGERPRINT_CHECK`, `PYRE_LLBC_STRICT`,
`PYRE_M73_BACKXLAT_TWIN_AUDIT`, `PYRE_M73_EMPTYTWIN_CENSUS`,
`PYRE_M73_LASTINSTR_AUDIT`, `PYRE_M73_MIDBODY_CARRY_AUDIT`,
`PYRE_MAJIT_STATS_ANCESTOR`, `PYRE_MAJIT_STATS_ROOT_ONLY`, `PYRE_MC_DIAG`,
`PYRE_MIR_FRAMESTATE_STRICT`, `PYRE_NO_JD1`, `PYRE_NO_UNROLL`,
`PYRE_PCMAP_AFTERRESIDUAL_AUDIT`, `PYRE_PCMAP_CONTAINING_AUDIT`,
`PYRE_PCMAP_RECIPE_RESULTCOLOR_AUDIT`, `PYRE_PCMAP_RESIDUAL_CENSUS`,
`PYRE_PORTAL_RCA`, `PYRE_PROBE_BH_STARTUP`, `PYRE_PROBE_SNAPSHOT`,
`PYRE_PROBE_SUBSCR`, `PYRE_PROFILE_PIPELINE`, `PYRE_QMUT_MAPDICT_FORCE`,
`PYRE_RERAISE_DIAG`, `PYRE_SIZE_SHELL_OWNERS`, `PYRE_SNAPSHOT_DIAG`,
`PYRE_WASM_DUMP_BAD_TRACE`, `PYRE_WASM_EXEC_TRACE`, `PYRE_WASM_FBW_CENSUS`,
`PYRE_WASM_GUARD_CENSUS`, `PYRE_WASM_JIT_STATS`, `PYRE_WASM_NO_CACHE`,
`PYRE_WASM_STARTUP_TRACE`.

## Summary

| bucket | count |
|---|---|
| retired (§1 + §1b + §1c + §1d parity pass) | 5 + 4 + 11 + 1 |
| not gates (identifiers) | 12 |
| dead (no read site) | 10 |
| live default-ON, kept until epic closes | 10 |
| diagnostics (OFF) | ~34 |
| default-OFF experiments (all keep — adoption targets) | 3 |
| config / value / master | ~17 |
| test harness | 1 |
