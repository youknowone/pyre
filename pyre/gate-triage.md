# PYRE_* gate triage

**Status**: living record. WS4 deliverable of `rework.md` (finding F5, gate
debt). Audited 2026-07-05 on branch `pc-map` by reading the actual
read-expression at every source site (four census passes). Polarity rule:
`is_some()`/`=="1"`/`.unwrap_or(false)` → default **OFF**; `is_none()` /
`!= Ok("0")` / `.unwrap_or(true)` → default **ON**; a parse of
number/path/list/mode → **VALUE** (config, not a boolean gate).
Re-audited 2026-07-18 on branch `nbody`.

The charter (§3.6, A7) says a gate is a staging area, not a home: each live
default-ON experiment gate is kept only until its epic closes, then its OFF
path is deleted and the gate retired. This table is the standing list of what
to retire and when.

**This file records the gates that are live now.** A gate whose reader is
deleted leaves the file with the reader; the retirement history the audits
accumulated was removed on 2026-08-20, and `git log -p pyre/gate-triage.md` is
where a retired gate's polarity, epic and removal date are read from. The two
brakes in `pyre/pyrex/tests/gate_triage_complete.rs` compare live names against
live read sites, so nothing here depends on a retirement row surviving. The
surviving sections keep their original numbers — §1, §3 and §5b are simply
gone — so references written against them still resolve.

## Headline

The raw `rg 'PYRE_[A-Z0-9_]+'` count (~119) overstates the debt. **~20 of the
matches are not env gates at all** (Rust consts, macro-generated identifiers,
runtime symbols, or comment-only dead references).

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

## §4 — Live default-ON gates KEPT (retire when the epic closes)

Each is default-ON but still a load-bearing kill switch for an open rework; its
OFF path is a needed safety net. Retire at the listed trigger (A7).

| var | subsystem | retire when |
|---|---|---|
| MAJIT_MIR_FRAMESTATE | framestate-threaded MIR lowering | MIR front-end #176/#181/#346 |
| MAJIT_GC_ITEMSBLOCK, PYRE_GC_PREBUILT_REMEMBER, PYRE_GC_INTERP, PYRE_GC_INTERP_COLLECT | GC-managed items / prebuilt minor-skip / interpreter allocation + collect rollback | WS3 / #355 / F3 GC rework |
| MAJIT_CL_NO_CLOSING_JUMP | cranelift attached-loop closing jump | #245 cranelift perf (explicit rollback hatch) |
| PYRE_NO_BINOP_REWIND | rewind at the `BINARY_OP` / `COMPARE_OP` dunder entry | #1526 binop-dunder inline (explicit rollback hatch) |

`PYRE_GC_INTERP` is default-ON on every target. Its OFF path still selects the
unmanaged `malloc_typed` stepping-stone allocation and remains a rollback hatch
until translated shadow-stack roots make the ordinary moving-nursery path safe.

`PYRE_NO_BINOP_REWIND` puts that entry's admission back on the whole-body
`Clean` verdict, which is the only bar it had before it had a rewind. It is the
control every reading of the entry is taken against, and both wrong answers
that shaped the admission were attributed with it: the one that showed a
delegating body aborts inside a nested sub-walk, and the one that showed no
static test can tell a committing body from the call-free body this entry
exists to admit. What replaced those tests is a refusal at the first residual
that could commit, taken before it runs. The gate goes when the #1526 entry
closes.

## §5 — Other live gates (not removal targets by the "already-ON" criterion)

Kept as-is; listed for completeness.

- **Diagnostics (~35, default-OFF)** — print/log/dump/probe/assert only; tooling,
  not experiments: `PYRE_FBW_DEBUG_ABORT`, `_INLINE_DIAG`, `_MF_DIAG`,
  `_STRICT_DIAG`, `PYRE_WALK_PERFN_JITCODE`, `PYRE_DUMP_PERFN_JITCODE`,
  `PYRE_P2_DIAG`, `PYRE_PCDEP_VALIDATE`, `MAJIT_MIR_FRAMESTATE_DEBUG`,
  `_FRAMESTATE_STRICT`, `MAJIT_MIR_FRONTEND_DEBUG`, `PYRE_VSTACK_DIAG`,
  `PYRE_PROBE_AUTHORITATIVE`, `_BH_STARTUP`, `_SNAPSHOT`, `_SUBSCR`,
  `MAJIT_S9_PROBE`, `MAJIT_PROFILE_DRAIN`, `_PIPELINE`, `PYRE_MFRAME_DIAG`,
  `MAJIT_RTYPER_VERBOSE`, `MAJIT_JTRANSFORM_SHADOW`, `PYRE_DIAG124C`, `_51C`,
  `_GIN`, `_INLINE_RECOG`, `PYRE_WASM_DUMP_ALL_TRACES`, `_DUMP_BAD_TRACE`,
  `_EXEC_TRACE`, `_JIT_STATS`, `PYRE_INTERP_RETURN_LOG`, `MAJIT_NBODY_DEBUG`,
  `PYRE_DEBUG_CALL`, `PYRE_DEBUG_CLASS`, `PYRE_DESCR_DEMAND`,
  `PYRE_CENSUS_HISTOGRAM`, `PYRE_REGEX_LENGTHS`.
  `PYRE_DESCR_DEMAND` records the distinct dense descriptor indices a run
  actually resolves, so the per-index pool loader can be measured against the
  pool size; the resolve path reads it through a `OnceLock` and pays nothing
  when it is unset. It is a measurement probe with no ON behaviour to graduate,
  so it has no epic — delete it with the demand counter itself once the pool's
  working set is settled.
  `PYRE_CENSUS_HISTOGRAM` prints the regex allocation census by exact request
  size. It is disabled by default and retires with the regex deopt-allocation
  investigation that consumes the histogram.
  `PYRE_REGEX_LENGTHS` selects the regex example input lengths used by that
  investigation. It is disabled by default and retires with the same
  measurement-only tooling.
- **Default-OFF experiments (0)** — every gate this bucket once held has had
  its reader and its ON path deleted, the last of them when the `LIST_APPEND`
  admission it gated became unconditional. The live default-OFF arms left in
  §6a2 are wasm A/Bs, kept as the switched-off side of a one-binary
  comparison rather than as experiments.
- **Config / value / master switches (~16)** — tuning, paths, modes; keep:
  `MAJIT_MIR_FRONTEND_LLBC`, `PYRE_WASM_ENGINE`, `_FUEL`, `_MODULE`, `_NO_CACHE`,
  `PYRE_GC_INTERP`, `PYRE_JIT`, `PYRE_NO_JIT`, `PYRE_STDLIB`,
  `PYRE_CHECK_PYPY3`, `PYRE_CHECK_PYTHON3`, `PYRE_SANDBOX_NO_SECCOMP`,
  `PYRE_SHARED_BUILD`, `PYRE_SYNTH_PYPY`, `PYRE_SYNTH_PYRE`, `PYRE_SYNTH_PYTHON`.
- **Test harness (1)**: `MAJIT_MIR_STRESS_LLBC`.

## §6 — The 66 gates the audits never listed (2026-08-07)

The hand audits above enumerated what they were looking at. Measured against the
tree instead, **66 of the 105 live gates had no entry anywhere in this file** —
the table was 63% empty, because nothing failed when a new gate skipped it.
`pyre/pyrex/tests/gate_triage_complete.rs` is now that failure: a `PYRE_*` read
with no entry here fails `cargo test`.

⚠ That reader is line-based, and one rule bites anyone adding prose: a line
whose text contains the past participle of "retire" marks **its first** `PYRE_*`
token as a closed subject, wherever in the file the line sits — so a sentence
that names a live gate and that word together silently un-documents the gate,
and the test then reports it as missing rather than as mis-parsed. Keep the
word and the name on separate lines.

The counts to quote, distinguished:

| count | value |
|---|---|
| distinct names read from the environment | **111** |
| — of those, read from Rust | 105 |
| — read only from the harness Python | 6 |
| (file, name) read pairs | 141 |
| **live gates that were absent from this file** | **66** |
| names still listed live with no read site left (retire) | 0 |

```sh
{ git ls-files '*.rs' ':!pyre/pyrex/tests/gate_triage_complete.rs'; \
  git ls-files 'pyre/*.py' 'pyre/**/*.py' 'scripts/*.py'; } \
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

**A retirement row documents nothing, wherever it sits.** §2 names things that
were never env vars, so no name in it counts. Section granularity alone cannot
carry the rule, though — a retirement note can sit inside a section whose
heading reads live — so any *row* saying "retired" is skipped as well, and its
subject stays skipped wherever else the file writes it. Re-introducing a reader
for a retired gate therefore fails the brake rather than passing on the strength
of its own obituary.

Polarity below follows this file's rule, with one correction it needed: an
`is_none()` whose value *is* the enable flag means default **ON**, but an
`if …is_none() { return; }` early-return guard means the thing is default
**OFF**. Three diagnostics (`PYRE_DESCR_SPELLING_GATE`, `PYRE_GC_DIAG`,
`PYRE_MC_DIAG`) read as ON under the unqualified rule and are OFF in fact.

### §6a — Live default-ON (6): the removal targets

| gate | what is ON by default | retire when |
|---|---|---|
| PYRE_JD1_NO_ENTER | entering the compiled jd1 loop directly rather than leaving the drain to the interpreter caller.  ON as a gate, but it decides nothing until `PYRE_JD1=1` arms jd1 (§6a2) | with `PYRE_JD1` |
| PYRE_WALKABORT_OFF | the non-carrier walk-abort leg (`trace.rs walk_abort_leg_enabled`) | kept deliberately: the leg commits irrevocably once the blackhole runs, so it is the one-binary A/B for the bug class it sits in |
| PYRE_WASM_BRIDGE_PARAMS | a wasm guard passing its fail args to the bridge as call parameters (`lib.rs bridge_params_enabled`); `=0`/`false`/`off` restores the jitframe spill crossing | the wasm trace-crossing epic closes; until then it is the one-binary A/B for the crossing shape |
| PYRE_WASM_INLINE_BRIDGE | merging a loop-closing bridge's ops into the module of the loop it guards into, so `guard → bridge → loop` becomes a `br` (`lib.rs inline_bridge_enabled`); `=0`/`false`/`off` restores the separate bridge module | the wasm trace-crossing epic closes; until then it is the one-binary A/B for the crossing shape |
| PYRE_WASM_FULL_TEARDOWN | skipping the ~0.2s wasm engine teardown at exit; setting it restores the drops for leak diagnostics | when teardown stops being the dominant fixed startup tax |
| PYRE_FBW_NO_ADOPT_RESIDUAL_LOCALS | reading back the fastlocals a residual wrote to the frame, whether or not it forced, as a recorded `GETARRAYITEM_GC_R` off `locals_cells_stack_w` (`residual_call.rs adopt_residual_locals_writes`); setting it restores the walk that keeps the box it held before the call and so loses the write | when the walk reads a local through a channel a residual cannot leave stale; until then this is the one-binary control that keeps the defect demonstrable, and the parity fixture's two arms (a forcing call, and an inlined callee whose store forces nothing) are only separable with it |

### §6a2 — Default-OFF experiments (6)

Kept as the switched-off arm of a one-binary comparison, not as latent
defaults.  Bridge inlining reaches module replacement on its own, so
`PYRE_WASM_REEMIT` adds only the one-shot rebuild-with-unchanged-content that
exercises the replacement machinery by itself.

`PYRE_JD1` is off for a third reason: the arm is incomplete rather than
wrong or unproven.  pyre drives jd1 through the same `MetaInterp.tracing`
slot as the bytecode portal, so while a residual `next()` runs an
arbitrarily large Python computation the shared tracing flag suppresses
every jd0 merge point the generator body reaches.  It stays dormant until
it has RPython's independent recursive-portal behavior.

`PYRE_FBW_INLINE_POISON` is off because its ON arm is known wrong, not merely
unproven: the replay scan reports the pcs it objects to instead of collapsing
them to one verdict, and the walk refuses on arriving at one, but that refusal
denies the callee for the rest of the thread's tracing and lands wherever the
walk happens to be. On the synthetic corpus it reaches a poisoned pc on 47 of
451 benches, and two of those answer wrong because the refusal follows an
executed effect. The scan and the enforcement stay wired so the arm that the
fix has to make sound can be measured against the collapsed verdict from one
build.

| gate | what turning it ON does | retire when |
|---|---|---|
| PYRE_WASM_REEMIT | re-emits a compiled loop's wasm module into its own table slot once, on the first bridge installed against it | when the replacement path no longer needs an isolated arm |
| PYRE_GUARD_RESUME_PC | prints the coordinate every walker-emitted guard resumes at (`resume_snapshot.rs guard_resume_pc_probe_enabled`); a guard whose `py_pc` is not the opcode it was emitted under re-executes the wrong bytecode on deopt, which reads as a livelock or a corrupted local rather than as a crash | the resume coordinate is covered by an ordinary test |
| PYRE_PORTAL_SPLIT | registers jd0 against the `warmspot.py split_graph_and_record_jitdriver` copy split immediately before `jit_merge_point`, instead of the unsplit `eval_loop_jit` graph; `=1` arms it and the prepass cache key includes the value | when the split portal path is the default and the unsplit registration arm is deleted |
| PYRE_WASM_INLINE_NONHEADER | admits an inlined region whose closing JUMP names a resumable LABEL other than the loop header AND whose source guard is in the LOOP BODY (`lib.rs inline_nonheader_enabled`); `=1`/`true`/`on` arms it.  The preamble-sourced half of that class takes a different placement — blocks outside the header `loop`, body past its `end` — and is admitted unconditionally, so this flag now covers only the body-sourced half.  Arming it removes 49.4M of the 257.3M cross-module crossings on the 81 fixtures that reach the decline and buys 0.74x/0.67x on two of them.  The `spectral_norm` loss the retirement condition below was written against no longer reproduces: its two regions are deferred and their bridges never reach the trip count, so the flag leaves its crossings and its merges alike untouched.  Across 536 bench fixtures, priced at the measured 0.67 ms per module + 0.493 ms/KB of cranelift and 4.3 ns per crossing, arming it models as 105.7 ms cheaper — four fixtures worth 188.6 ms against twenty-odd worth 83 ms, the worst being `kept_stack_deep_var_shortcircuit` at 53KB of added module for 40k crossings | the +18 ops per non-failing iteration it levies on the owner's fall-through is paid back on the fixtures it admits, measured on a machine quiet enough to grade wall clock rather than modelled |
| PYRE_WASM_COMPILE_CENSUS | reports every cranelift compile of a trace module separately (`main.rs jit_compile_trace`) — the bytes handed over, the wall time it took, and whether the request was a first compile or the re-emission of an owner that took a merge.  The stats line carries only the run's totals, which cannot separate a re-emission's cost from a first compile's nor say whether the per-module cost is linear in the bytes | trace compilation stops being on the critical path, or the two questions are answered and the answers stop moving |
| PYRE_WASM_INLINE_TRIP_BYTES | `=N` prices a deferred inline merge at N bridge entries per byte of the module the merge re-emits (`lib.rs inline_trip_threshold_for`), the guest's flat `INLINE_TRIP_THRESHOLD` staying as the floor; unset leaves the built-in `DEFAULT_INLINE_TRIP_BYTES_FACTOR`, and `=0` restores the flat threshold as the whole rule.  A merge re-emits its owner whole, so its cranelift cost scales with the owner's size while the crossings it removes scale with the standing bridge's entries — the flat threshold reads only the second.  Exists so the conversion between the two rates can be re-swept on ONE binary, the guest having no environment to read it from | the corpus stops disagreeing about the value, or the merge decision stops being a single scalar |
| PYRE_FBW_INLINE_POISON | admits a callee the replay scan declined and refuses at the scan's poisoned pcs during the walk (`diag.rs fbw_inline_poison_enabled`) | when a refusal that follows an executed effect has a resume leg that neither repeats it nor drops it |
| PYRE_JD1 | arms the jd1 (`unpackiterable_driver`) compiled-loop experiment — `eval.rs jd1_experiment_enabled` is `PYRE_JD1 == "1"`, so nothing else turns it on.  `PYRE_NO_JD1`, `PYRE_JD1=0` and the master JIT off-switches (`PYRE_NO_JIT`, `PYRE_JIT=0`) each force it back off | the jd1 experiment concludes |

### §6b — VALUE knobs (17): config, not gates

`PYRE_FBW_MULTIFRAME_DEPTH`,
`PYRE_FBW_NO_SPECIALIZE`, `PYRE_JD1_THRESHOLD`,
`PYRE_PCMAP_RECIPE_RESULTCOLOR_AUDIT_PROBE`,
`PYRE_PORTAL_METATRACE_ENTRY`, `PYRE_PORTAL_METATRACE_SKIP`,
`PYRE_CENSUS_TRACE_SIZE`, `PYRE_CENSUS_TRACE_SKIP`,
`MAJIT_DTRACE_CONST_FROM`, `MAJIT_DTRACE_CONST_TO`,
`MAJIT_TRACE_CALL_DIAG`, `MAJIT_TRACE_OPS_DIAG`,
`PYRE_WASM_FORCE_CA_TERMINAL_DECLINE`, `PYRE_WASM_FUEL`,
`PYRE_WASM_GUEST_PROFILE`, `PYRE_WASM_MODULE`.

`PYRE_CENSUS_TRACE_SIZE` enables the alloc-census example's bounded
backtrace attribution for one exact allocation size, and is disabled when
unset. `PYRE_CENSUS_TRACE_SKIP` is its optional nonnegative sample offset and
defaults to zero. Both are measurement inputs, not runtime experiments.

`PYRE_FBW_NO_SPECIALIZE` is the one entry here that changes behaviour rather
than reporting it: its comma-separated selectors (or the reserved `all`) turn
off that many of the 77 trace-time specialization rows (the `spec_folds!`
invocation at `jitcode_dispatch/diag.rs:342-421`; count them there rather
than trusting this sentence), and an unset variable suppresses none.  Not all
75 are hand-written: `subscr_tuple_descent`, `unary_invert_descent` and
`unary_negative_descent` name orthodox sub-walks of the interpreter's own
body, and a row is what lets one be suppressed and A/B'd like any other. It is a measurement instrument — suppressing a fold is how the descent
wall behind it is made to print — so it retires with the folds it selects,
not before them.
`PYRE_FBW_SPEC_CENSUS` in §6c is its read-only half: the per-fold
consulted/fired tallies. `PYRE_WASM_SPEC_CENSUS` is that same readout on the
wasm backend, where the guest reads no environment and the runner has to arm
it through the `pyre_fbw_spec_census_enable` export instead.

`PYRE_PORTAL_METATRACE_ENTRY` selects where the one-shot jd0 portal probe
starts: `merge` (the default) seeds the merge-point registers and `start`
enters at pc 0 with the portal's declared arguments.  The numeric
`PYRE_PORTAL_METATRACE_SKIP` value selects how many cached-loop back-edges the
probe passes before firing; it defaults to zero.  Neither has an effect unless
the probe in §6c is enabled.

### §6c — Default-OFF diagnostics, censuses and probes (76): keep, cost nothing

Deleting one of these environment reads does not change behavior when the
variable is unset. They remain listed so diagnostics are not mistaken for dead
configuration.

`PYRE_ALLOCSITES`, `PYRE_BH_NULL_ARG`, `PYRE_BRIDGE_LATCH_AUDIT`,
`MAJIT_CALLEE_RCA`, `PYRE_CATCH_LIVE_CENSUS`,
`PYRE_CELL_CENSUS`, `PYRE_CHECK_INHERIT_ENV`,
`PYRE_DESCR_SPELLING_GATE`,
`PYRE_DEOPT_PROBE`, `PYRE_DIAG_51C`, `PYRE_DIAG_GIN`, `PYRE_DIAG_INLINE_RECOG`,
`MAJIT_DETERMINISM_TRACE`, `MAJIT_DTRACE_CONST_BT`,
`MAJIT_DYNASM_EXEC_DIAG`, `PYRE_FBW_CENSUS`, `PYRE_FBW_DEPTH_CENSUS`,
`PYRE_FBW_DESCENT_SCAN_OFF`, `PYRE_FBW_INLINE_DIAG`,
`PYRE_FBW_LOOPBODY_SCAN_FULL`, `PYRE_FBW_LOOPBODY_SCAN_LOOP_ONLY`,
`PYRE_FBW_MF_DIAG`, `PYRE_FBW_REPLAY_DIRTY_BODY`, `PYRE_FBW_SPEC_CENSUS`,
`PYRE_FBW_STRICT_DIAG`,
`PYRE_FIELD_IDENTITY_CENSUS`,
`PYRE_FORITER_INFLIGHT_CENSUS`, `PYRE_FOR_ITER_GATE_DIAG`,
`PYRE_GC_DIAG`, `MAJIT_GC_FREELIST_DIAG`, `PYRE_GC_SIZE_AUDIT`,
`PYRE_GEN_CENSUS`, `PYRE_GEN_ENTRY_DIAG`,
`PYRE_JD1_DEBUG`, `PYRE_JD1_DUMP`,
`PYRE_LB_SITE`, `PYRE_LLBC_SKIP_FINGERPRINT_CHECK`, `PYRE_LOOP_CENSUS`,
`PYRE_M73_BACKXLAT_TWIN_AUDIT`, `PYRE_M73_EMPTYTWIN_CENSUS`,
`PYRE_M73_LASTINSTR_AUDIT`, `PYRE_M73_MIDBODY_CARRY_AUDIT`,
`PYRE_MAJIT_STATS_ANCESTOR`, `PYRE_MAJIT_STATS_ROOT_ONLY`, `PYRE_MC_DIAG`,
`MAJIT_MIR_FRAMESTATE_STRICT`, `PYRE_NO_JD1`, `MAJIT_NO_UNROLL`,
`PYRE_PCMAP_AFTERRESIDUAL_AUDIT`, `PYRE_PCMAP_CONTAINING_AUDIT`,
`PYRE_PCMAP_RECIPE_RESULTCOLOR_AUDIT`, `PYRE_PCMAP_RESIDUAL_CENSUS`,
`MAJIT_PORTAL_RCA`, `PYRE_PORTAL_METATRACE`, `PYRE_PROBE_BH_STARTUP`,
`PYRE_PROBE_SNAPSHOT`,
`MAJIT_PROBE_SUBSCR`, `MAJIT_PROFILE_PIPELINE`,
`PYRE_RERAISE_DIAG`, `MAJIT_SIZE_SHELL_OWNERS`, `PYRE_SNAPSHOT_DIAG`,
`PYRE_UNJOURNALED_SITE`,
`PYRE_VSTACK_EXACT_AUDIT`, `PYRE_VSTACK_KEEP_REORDER`, `PYRE_VSTACK_NO_EXACT`,
`PYRE_WASM_DUMP_BAD_TRACE`, `PYRE_WASM_EXEC_TRACE`, `PYRE_WASM_FBW_CENSUS`,
`PYRE_WASM_GUARD_CENSUS`, `PYRE_WASM_JIT_STATS`, `PYRE_WASM_CALL_HIST`,
`PYRE_WASM_NO_CACHE`, `PYRE_WASM_SPEC_CENSUS`, `PYRE_WASM_STARTUP_TRACE`,
`PYRE_WASM_TRACE_ENTRY_CENSUS`.

`PYRE_WASM_JIT_STATS` prints its readout for any value. The one value it
reads is `nofuel`, which asks for the readout without wasmtime's fuel metering.
Metering charges every guest instruction, which makes the host cranelift
compile that `compile_ms` and `compile_bytes` measure slower than it is in a
production run; `nofuel` is how those two fields are read. `wasm_ops` is the
fuel subtraction and reports -1 under it.

`PYRE_FBW_DESCENT_SCAN_OFF` turns off the descent's un-lowered-helper scan
(`descent_unlowered_helper_scan_enabled`), so the walker descends into a
builtin body that holds a symbolic-fnaddr residual call and aborts at the call
instead of declining before the descent starts. Same shape and same reason as
`PYRE_WALKABORT_OFF`: the scan decides whether a descent happens at all and its
cost is invisible in output, so weighing the conservatism against its price
needs one binary and one variable. It retires when the scan does.

`PYRE_GC_SIZE_AUDIT` makes `finish_alloc_in_oldgen` panic, with a captured
backtrace, when the block it just carved is smaller than the declared size of
the type its header names. It is the discriminator between a header stamped
wrong at allocation and a genuine use-after-free: the collector's own
`GC BUG` panics fire a whole major later and name the freed object, not the
site that mis-sized it. Varsize and unregistered type ids are skipped, so it
answers only where the declared size is authoritative. It retires if the
check becomes unconditional.

`PYRE_ALLOCSITES` enables stack attribution in the standalone `allocsites`
example; it is unset by default. Its `AFTER`, `BUDGET`, `EVERY`, and `ROWS`
value knobs bound the capture window, sampling rate, and report size. This is a
diagnostic tool rather than a temporary runtime experiment, so it retires only
if the example itself is removed.

`PYRE_FBW_REPLAY_DIRTY_BODY` is a sub-knob of `PYRE_FBW_INLINE_DIAG` rather
than a gate of its own: `replay_safety_dump_body` returns unless both are set,
so setting it alone prints nothing. It lists each callee body as it is scanned,
which is what lets the `pc` on a following `[replay-dirty]` line be matched to
an op. It goes with the inline diagnostic it extends.

`PYRE_VSTACK_NO_EXACT` and `PYRE_VSTACK_KEEP_REORDER` are A/B switches over the
walk-level operand-stack mirror, each restoring the behaviour its default
replaced: resolving the mirror's Python-PC coordinate from the floor tier rather
than from the per-emission segmentation, and leaving an armed out-of-order
region in place across a mirror re-seed. They exist so each switchover stays
measurable on one binary — any env var of any name moves the allocation layout,
so a knob and its control must live in the same build. Each goes when the
behaviour it restores has no plausible reader left to compare against.
`PYRE_VSTACK_EXACT_AUDIT` dumps that segmentation table at build time and, on
the walk side, the coordinate it yields; it is a report with no ON behaviour.

`PYRE_UNJOURNALED_SITE` names the site and the call opcode behind the
walk-level "this walk still owes an effect" flag, which six call sites set and
the flag itself keeps no provenance for. It is the second half of a two-step
read: `PYRE_FBW_CENSUS` finds the walks that ended `committed=false` with
unrecoverable effects, and this one says which residual decline put them there.
It goes when the flag carries its own provenance.

`PYRE_PORTAL_METATRACE` drives one Stage-0 `JitCodeMachine` walk over the
build-time jd0 portal jitcode after the selected cached-loop back-edge and
prints the `[jd0-mt]` summary.  It is unset by default and exists to inspect
the portal split and entry seeding; it goes with that investigation once the
ordinary warmspot path owns the same coverage.

`PYRE_LOOP_CENSUS` prints one `[loop-census] <arm> <name>` line per compiled
trace, naming it through `get_printable_location` — the JitDriver green-key
hook ported for parity in `pyre-jit`. Unlike the rest of this section it has a
gate consumer, not just a reader: `check.py` sets it on every selfcheck run and
grades the run against each fixture's `# pyre-check: selfcheck-compiles=`
header, so retiring it would silently un-gate those fixtures. It goes when a
compiled trace carries its own identity somewhere a gate can read without a
diagnostic env var.

`PYRE_CHECK_INHERIT_ENV` is the other odd one: an A/B switch, not a report.
`check.py` starts a benchmark child from an allowlisted environment because the
inherited one is startup allocation and moves `guard_failures`; setting the
gate restores the whole-environment copy, so the size of that effect stays
measurable on one binary. It goes when the allowlist stops being the thing
under measurement.

### §6d — Default-ON safety controls

| gate | behavior | retirement condition |
|---|---|---|
| `PYRE_LLBC_STRICT` | treats stale frozen LLBC artifacts as an error; setting it to `0` demotes the error to a warning | retain while the build consumes frozen LLBC artifacts |


## §7 — Additional runtime diagnostics

| gate | default polarity | what it gates / retirement condition |
|---|---|---|
| `PYRE_PROBE14` | OFF | reports discarded reference-constant relocations; retire when relocation preservation is covered by ordinary tests |
| `PYRE_GC_SIZE_AUDIT` | OFF | panics when a block is stamped with a type id whose declared payload is larger than the block's own extent, at the allocation that stamps it rather than in whichever later collection reads the neighbouring block as a field (varsize types are exempt); retire when every allocator derives the size from the type id it stamps, so the two cannot disagree |
| `PYRE_GC_GATE_BASE` | VALUE | names the upstream commit `scripts/check-gc-root-brackets.py` measured its numbers over, so a backlog raised by code the baseline never saw is reported rather than charged to the branch; the workflow supplies it because a CI checkout is shallow, holds no `main` ref and has its merge commit's parent list truncated away, leaving nothing in the repository able to answer; retire when the gate's job checks out enough history for `git merge-base` to name the base itself |
| `PYRE_EXIT_FRAME_DIAG` | OFF | prints one line per `exit_frame_with_exception` delivery to a frame's own exception table, naming the site and the verdict `exit_frame_handler_needs_unwritten_stack` reached there, passes included so a refusal is a share of something; retire when the handler search moves inside the trace and the two delivery sites become one |

## Summary

| bucket | count |
|---|---|
| not gates (identifiers) | 12 |
| live default-ON, kept until epic closes | 8 |
| diagnostics (OFF) | ~35 |
| default-OFF experiments | 2 |
| config / value / master | ~17 |
| test harness | 1 |
