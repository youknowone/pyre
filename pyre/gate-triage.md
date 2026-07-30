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
Still kept: `PYRE_CARRIER_EXC_RESUME` (default-off; threads the guard-failure exception
into the bridge sym for the depth-2 carrier exception-resume slice #343/#126 —
inert until validated).  Two parity gaps were listed here as pre-flip work.  The
`bridge_guard_exc` GC-rooting is closed by §1e — which also measures the seed
site as **reachable** (170 bridge-route guard failures carry a live exception),
so this gate is a live adoption target rather than an inert one.

The second is still open, and the row described it as "the unconditional
`execute_ll_raised` exception assign", which is not what the divergence is.

pyre's standing-exception maintenance for an exception-guard bridge lives in
`seed_bridge_standing_exception_from_current` (`state.rs`), which is **not
gated** and already mirrors upstream's branch: it assigns `last_exc_value` /
`last_exc_box` when it finds an exception, and clears all four exception slots
when it does not (`_prepare_exception_resumption`'s
`else: clear_exception()`).  The divergence is the **source**.  Upstream takes it
from `cpu.grab_exc_value(deadframe)` — the exception the failing guard carried.
pyre takes it from `sym.current_exc_value`, falling back to
`get_current_exception()` — the *execution context's* current exception, which is
the `sys.exc_info()` mirror, a different slot with different lifetime rules.

`PYRE_CARRIER_EXC_RESUME` is a **back-channel into that function**: its only
effect is to write `guard_exc` into `current_exc_value` beforehand so the ungated
code picks it up.  Hence the `is_null` conjunct — it exists to avoid clobbering a
live `sys.exc_info` value, which also means the injection is suppressed exactly
when the EC already holds an exception.  That is why forcing the gate on measures
as a no-op: **dynasm 336/336 with the gate forced on**, correctness results
matching the default run, and the seven live-exception producers of §1e among
them, despite the seed site being entered 170 times.

So "inert until validated" should read **inert because the guard's exception
reaches `last_exc_value` only through a slot it does not belong in**.  A green
corpus under the gate is not evidence about the gate.  Two further deltas to
settle before any flip, both in that function: it early-returns when
`last_exc_box` is already set, and it sets `class_of_last_exc_is_const = true`,
whereas the `_prepare_exception_resumption` path reaches `execute_ll_raised` with
the default `constant=False`.

## §1e — The grabbed guard exception is rooted for the whole handoff (2026-07-27)

`bridge_guard_exc` was booked as a pre-flip gap for `PYRE_CARRIER_EXC_RESUME`.
It is **not gate-specific**: the same grabbed pointer drives the default
blackhole resume, so the gate never bounded the exposure.

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
the last row are exactly the `bridge_guard_exc` read this section is about — the
`PYRE_CARRIER_EXC_RESUME` seed site is **reachable**, not inert.  Seven benches
produce them: `inline_subwalk_property_mutates` and
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

## §1c — Retired since the 2026-07-05 audit (10): reader already deleted by a closed epic

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
| PYRE_FBW_MULTIFRAME | **ON** | **FLIPPED default-ON** — the ON path is the port and the adopt works; §1d's one remaining wrong answer (a `sys._getframe` that is itself the escaping residual reading the caller frame) was closed by `walker_ec_enter` / `walker_ec_leave` publishing the callee frame at the inlined-call push |
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
(`pyre/bench` + `pyre/bench/synth`) run under `PYRE_FBW_MULTIFRAME=1`.  The site
is reached in **3 benches** (`getframe_escape_flush_writethrough_regression`,
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
path.  With the comparison fixed, that fixture under `PYRE_FBW_MULTIFRAME=1`
reports **5 `BUILT multi-frame depth=2` and 5 `adopted multi-frame terminal`,
zero declines** (the other 5 escapes in the run have `inline_subwalk=false` and
take the single-frame arm, as before), and prints the same result as CPython and
PyPy.

**What the build still declines, and why the decline is right.**  Two shapes
reach the latch and are then refused by `capture_inline_parent_blackhole`
(`resume_snapshot.rs`): a caller with an exception handler around the inlined
call, and two nested inlined levels (depth 3).  Instrumented 2026-07-26, both
report the same cause — a ref color that is **live at the caller's post-call
coordinate holds `ConcreteValue::Null`**:

```
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

**The flip is blocked, and the blocker is a wrong answer, not a decline.**
Measured 2026-07-26.  The walker executes residuals **concretely** while an
inline push never runs the interpreter's call sequence, so `ec.topframeref`
still names the CALLER while an inlined callee body runs.  A `sys._getframe`
that is *itself* the escaping residual therefore reads the wrong frame at walk
time, and the adopt commits that answer where legacy escape/replay discards it:

```
_gf().f_code.co_name   -> "main",     not "leaf"
_gf(1).f_code.co_name  -> "<module>", not "main"     # one level too far up
_gf(1).f_locals["k"]   -> KeyError                   # same cause, seen through the argument
```

**One wrong iteration per multi-frame adopt** — 5 adopts, 5 wrong, in each part
of `synth/getframe_while_escaping_read_frame_identity`, which is the acceptance
test: it passes today (gate off, the default) and fails loudly if the gate is
flipped first.  This is *not* outer-locals staleness.  A `sys._getframe`
executed **after** the escape, inside the blackhole, is correct — the chain
publishes each level's frame as it runs — and an in-blackhole read of a caller
local mutated earlier in the same iteration was measured correct against CPython
and PyPy.  Closing it needs the inlined-call push to publish the callee frame on
the execution context, which is what the open `walker_ec_enter` / `walker_ec_leave`
work does; the `jit.virtual_ref` emit rides along with it.  So the original
decline comment was right that an inline-push `enter` is the prerequisite, and
wrong only about which check it gated.

One thing the ON path already fixes: with a side-effecting inlined callee under
a `while` loop that returns from inside the loop, the OFF path runs the callee's
side effect ~5.2k extra times (the recorded trace-abort double-run class) while
the adopt gives the exact count.

Everything else that was thought to block the flip has been measured and does
not: the full corpus is **336/336 with the gate on (dynasm) and 336/336 with it
off (cranelift)**, the blast radius is exactly `inline_subwalk = true` at a
vable escape (the latch is an `if`/`else if` whose single-frame arm requires
`!inline_subwalk`, so with the gate off that condition latches nothing and falls
to legacy escape/replay), and the two build-side declines above are correct.

Note the multi-frame latch is
nested inside `single_frame_blackhole_resume_enabled()`, so it also requires
`_BLACKHOLE_RESUME` to stay ON.  The pre-existing `[s2-gate]` eprintln (under
`PYRE_FBW_DEBUG_ABORT`) already reports `inline_subwalk` at that site.

**A second coverage benchmark, 2026-07-27.**
`synth/getframe_inline_subwalk_multiframe` (#798) reaches the latch with
`inline_subwalk=true` and drives `build_multi_frame_miframe` — under
`PYRE_FBW_MULTIFRAME=1 PYRE_FBW_DEBUG_ABORT=1` it prints 5 `[s2-gate]
inline_subwalk` lines each followed by `[s2-build-decline] BUILT multi-frame
depth=2`.  One `sys._getframe(1)` level does not get there; the chain needs a
residual level under the walked frame and an inlined level under that, so the
force has to reach two frames up.  Per-frame vable binding, outer-locals
materialization, and the `jit.virtual_ref` emit are therefore validatable now.

## §2 — Not gates (11): Rust identifiers, not env vars

The audit regex matched non-env identifiers. These are real code; **do not
delete, do not count as gates.**

- `PYRE_STR_DESCR`, `PYRE_STR_BYTE_LEN_DESCR`, `PYRE_UNICODE_DESCR`,
  `PYRE_UNICODE_LEN_DESCR` — field-descriptor `const`s (`pyre-jit-trace/src/pyre_cpu.rs`)
- `PYRE_CLASS_DESCRIPTOR` — macro-built identifier `W_{}_PYRE_CLASS_DESCRIPTOR` (`pyre-macros`)
- `PYRE_PARAM_NAMES`, `PYRE_PARAM_REQUIRED` — macro `const __PYRE_PARAM_*` (`pyre-macros`)
- `PYRE_JIT_GRAPH_MODULES` — compile-time `const &[&str]` module manifest (`generated.rs`)
- `PYRE_REF_OPAQUE` — `OpaqueType::gc("PYRE_REF_OPAQUE")` type label (`annotator/builtin.rs`)
- `PYRE_JIT_DISABLED` — a `OnceLock<bool>` cache name holding the `PYRE_JIT==0` result (`pyre-jit/src/eval.rs`); the env var is `PYRE_JIT`
- `PYRE_STACKTOOBIG` — `pub static PyreStackTooBig` runtime symbol (`stack_check.rs`)

## §3 — Dead (10): no env read site

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

## §4 — Live default-ON gates KEPT (retire when the epic closes)

Each is default-ON but still a load-bearing kill switch for an open rework; its
OFF path is a needed safety net. Retire at the listed trigger (A7).

| var | subsystem | retire when |
|---|---|---|
| PYRE_FBW_BLACKHOLE_RESUME | single-frame resume-past-escape (#754) | flipped default-ON 2026-07-25; retirement was conditioned on the multi-frame twin (`_MULTIFRAME`) flipping, and **that condition is now met** — the twin flipped once `walker_ec_enter` / `walker_ec_leave` closed the escaping `sys._getframe` identity answer. Both gates are now default-ON with the multi-frame latch nested inside `single_frame_blackhole_resume_enabled()`, so retiring the pair (deleting both readers and the OFF paths) is the next step for this row |
| PYRE_TWO_PHASE_RTYPE, PYRE_TUPLE_PER_SHAPE_CLASSDEF | rtyper prepass / per-shape tuple classdef | WS2 / #346 rtyper epic |
| PYRE_ORIGINAL_BOXES | greens++reds original_boxes index shape | box-identity #202 / resume F1 |
| PYRE_MIR_FRAMESTATE | framestate-threaded MIR lowering | MIR front-end #176/#181/#346 |
| PYRE_GC_ITEMSBLOCK, PYRE_GC_PREBUILT_REMEMBER, PYRE_GC_INTERP_COLLECT | GC-managed items / prebuilt minor-skip / interp collect A/B | WS3 / #355 / F3 GC rework |
| PYRE_CL_NO_CLOSING_JUMP | cranelift attached-loop closing jump | #245 cranelift perf (explicit rollback hatch) |

`PYRE_GC_INTERP` is default-ON on wasm32 only (`unwrap_or(cfg!(wasm32))`),
default-OFF on native — not a clean removal candidate.

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
- **Default-OFF experiments (2 remaining)** — triaged in §1b/§1c (4 retired
  in the 2026-07-05 pass, 8 retired since then; `PYRE_P2_DRAIN` retired with
  the framestack-walk deletion; `_VABLE_SCALAR_CA` retired 2026-07-25, see
  §1d).  Kept: `_CALLEE_VSTACK` (callee-local operand-stack mirror) and
  `PYRE_CARRIER_EXC_RESUME`.  For these the *ON* path is the unattested one, so
  they are adoption targets rather than retirement targets.
  `_BLACKHOLE_RESUME` graduated out of this bucket on 2026-07-25 (flipped
  default-ON, now in §4); `_MULTIFRAME` graduated the same way once
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
- **Config / value / master switches (~18)** — tuning, paths, modes; keep:
  `PYRE_FBW_REC_UNROLL`, `PYRE_WALKER_STORE_SUBSCR_FNADDR`,
  `PYRE_MIR_FRONTEND_LLBC`, `PYRE_WASM_ENGINE`, `_FUEL`, `_MODULE`, `_NO_CACHE`,
  `PYRE_GC_INTERP`, `PYRE_JIT`, `PYRE_NO_JIT`, `PYRE_STDLIB`,
  `PYRE_CHECK_PYPY3`, `PYRE_CHECK_PYTHON3`, `PYRE_SANDBOX_NO_SECCOMP`,
  `PYRE_SHARED_BUILD`, `PYRE_SYNTH_PYPY`, `_PYRE`, `_PYTHON`.
- **Test harness (1)**: `PYRE_MIR_STRESS_LLBC`.

## Summary

| bucket | count |
|---|---|
| retired (§1 + §1b + §1c + §1d parity pass) | 5 + 4 + 10 + 1 |
| not gates (identifiers) | 11 |
| dead (no read site) | 10 |
| live default-ON, kept until epic closes | 9 (+ `PYRE_GC_INTERP`, wasm32-only) |
| diagnostics (OFF) | ~34 |
| default-OFF experiments (all keep — adoption targets) | 3 |
| config / value / master | ~18 |
| test harness | 1 |
