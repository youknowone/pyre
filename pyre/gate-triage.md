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
| PYRE_MIR_FRAMESTATE | framestate-threaded MIR lowering | MIR front-end #176/#181/#346 |
| PYRE_GC_ITEMSBLOCK, PYRE_GC_PREBUILT_REMEMBER, PYRE_GC_INTERP, PYRE_GC_INTERP_COLLECT | GC-managed items / prebuilt minor-skip / interpreter allocation + collect rollback | WS3 / #355 / F3 GC rework |
| PYRE_CL_NO_CLOSING_JUMP | cranelift attached-loop closing jump | #245 cranelift perf (explicit rollback hatch) |

`PYRE_GC_INTERP` is default-ON on every target. Its OFF path still selects the
unmanaged `malloc_typed` stepping-stone allocation and remains a rollback hatch
until translated shadow-stack roots make the ordinary moving-nursery path safe.

## §5 — Other live gates (not removal targets by the "already-ON" criterion)

Kept as-is; listed for completeness.

- **Diagnostics (~35, default-OFF)** — print/log/dump/probe/assert only; tooling,
  not experiments: `PYRE_FBW_DEBUG_ABORT`, `_INLINE_DIAG`, `_MF_DIAG`,
  `_STRICT_DIAG`, `PYRE_WALK_PERFN_JITCODE`, `PYRE_DUMP_PERFN_JITCODE`,
  `PYRE_P2_DIAG`, `PYRE_PCDEP_VALIDATE`, `PYRE_MIR_FRAMESTATE_DEBUG`,
  `_FRAMESTATE_STRICT`, `PYRE_MIR_FRONTEND_DEBUG`, `PYRE_VSTACK_DIAG`,
  `PYRE_PROBE_AUTHORITATIVE`, `_BH_STARTUP`, `_SNAPSHOT`, `_SUBSCR`,
  `PYRE_S9_PROBE`, `PYRE_PROFILE_DRAIN`, `_PIPELINE`, `PYRE_MFRAME_DIAG`,
  `PYRE_RTYPER_VERBOSE`, `PYRE_JTRANSFORM_SHADOW`, `PYRE_DIAG124C`, `_51C`,
  `_GIN`, `_INLINE_RECOG`, `PYRE_WASM_DUMP_ALL_TRACES`, `_DUMP_BAD_TRACE`,
  `_EXEC_TRACE`, `_JIT_STATS`, `PYRE_INTERP_RETURN_LOG`, `PYRE_NBODY_DEBUG`,
  `PYRE_DEBUG_CALL`, `PYRE_DEBUG_CLASS`, `PYRE_DESCR_DEMAND`.
  `PYRE_DESCR_DEMAND` records the distinct dense descriptor indices a run
  actually resolves, so the per-index pool loader can be measured against the
  pool size; the resolve path reads it through a `OnceLock` and pays nothing
  when it is unset. It is a measurement probe with no ON behaviour to graduate,
  so it has no epic — delete it with the demand counter itself once the pool's
  working set is settled.
- **Default-OFF experiments (none remaining)** — every gate this bucket held
  has had its reader and its ON path deleted. The live default-OFF arms that
  remain are the two wasm re-emission A/Bs in §6a2, which are kept as the
  switched-off side of a one-binary comparison rather than as experiments
  waiting to graduate.
- **Config / value / master switches (~16)** — tuning, paths, modes; keep:
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

### §6a — Live default-ON (5): the removal targets

| gate | what is ON by default | retire when |
|---|---|---|
| PYRE_JD1 | the jd1 compiled-loop experiment (`eval.rs jd1_experiment_enabled`); `PYRE_NO_JD1` or `PYRE_JD1=0` turns it off, and no-JIT implies off | the jd1 experiment concludes |
| PYRE_JD1_NO_ENTER | entering the compiled jd1 loop directly rather than leaving the drain to the interpreter caller | with `PYRE_JD1` |
| PYRE_WALKABORT_OFF | the non-carrier walk-abort leg (`trace.rs walk_abort_leg_enabled`) | kept deliberately: the leg commits irrevocably once the blackhole runs, so it is the one-binary A/B for the bug class it sits in |
| PYRE_WASM_BRIDGE_PARAMS | a wasm guard passing its fail args to the bridge as call parameters (`lib.rs bridge_params_enabled`); `=0`/`false`/`off` restores the jitframe spill crossing | the wasm trace-crossing epic closes; until then it is the one-binary A/B for the crossing shape |
| PYRE_WASM_FULL_TEARDOWN | skipping the ~0.2s wasm engine teardown at exit; setting it restores the drops for leak diagnostics | when teardown stops being the dominant fixed startup tax |

### §6a2 — Default-OFF experiments (2): the wasm re-emission A/Bs

Both are measured to lose today and are kept as the switched-off arm of a
one-binary comparison, not as latent defaults.

| gate | what turning it ON does | retire when |
|---|---|---|
| PYRE_WASM_INLINE_BRIDGE | merges a bridge's ops into the loop module that guards into it, so `guard → bridge → loop` becomes a `br` | the wasm trace-crossing epic closes, or the shape is measured to win |
| PYRE_WASM_REEMIT | re-emits a compiled loop's wasm module into its own table slot, which is what lets an inlined bridge reach live code | with `PYRE_WASM_INLINE_BRIDGE` |

### §6b — VALUE knobs (12): config, not gates

`PYRE_FBW_MULTIFRAME_DEPTH`, `PYRE_FBW_NO_SPECIALIZE`, `PYRE_JD1_THRESHOLD`,
`PYRE_PCMAP_RECIPE_RESULTCOLOR_AUDIT_PROBE`,
`PYRE_DTRACE_CONST_FROM`, `PYRE_DTRACE_CONST_TO`,
`PYRE_TRACE_CALL_DIAG`, `PYRE_TRACE_OPS_DIAG`,
`PYRE_WASM_FORCE_CA_TERMINAL_DECLINE`, `PYRE_WASM_FUEL`,
`PYRE_WASM_GUEST_PROFILE`, `PYRE_WASM_MODULE`.

`PYRE_FBW_NO_SPECIALIZE` is the one entry here that changes behaviour rather
than reporting it: its comma-separated selectors (or the reserved `all`) turn
off that many of the 55 hand-written trace-time specialization rows, and an
unset variable suppresses none. It is a measurement instrument — suppressing a
fold is how the descent wall behind it is made to print — so it retires with
the folds it selects, not before them.
`PYRE_FBW_SPEC_CENSUS` in §6c is its read-only half: the per-fold
consulted/fired tallies.

### §6c — Default-OFF diagnostics, censuses and probes (70): keep, cost nothing

Each is inert unless set, so none is a removal target by this file's
already-ON criterion. They are listed so they cannot be missed again.

`PYRE_ALLOCSITES`, `PYRE_BH_NULL_ARG`, `PYRE_CALLEE_RCA`, `PYRE_CATCH_LIVE_CENSUS`,
`PYRE_CELL_CENSUS`, `PYRE_CHECK_INHERIT_ENV`,
`PYRE_DESCR_SPELLING_GATE`,
`PYRE_DEOPT_PROBE`, `PYRE_DIAG_51C`, `PYRE_DIAG_GIN`, `PYRE_DIAG_INLINE_RECOG`,
`PYRE_DETERMINISM_TRACE`, `PYRE_DTRACE_CONST_BT`,
`PYRE_DYNASM_EXEC_DIAG`, `PYRE_FBW_CENSUS`, `PYRE_FBW_DEPTH_CENSUS`,
`PYRE_FBW_INLINE_DIAG`,
`PYRE_FBW_LOOPBODY_SCAN_FULL`, `PYRE_FBW_LOOPBODY_SCAN_LOOP_ONLY`,
`PYRE_FBW_MF_DIAG`, `PYRE_FBW_REPLAY_DIRTY_BODY`, `PYRE_FBW_SPEC_CENSUS`,
`PYRE_FBW_STRICT_DIAG`,
`PYRE_FIELD_IDENTITY_CENSUS`,
`PYRE_FORITER_INFLIGHT_CENSUS`, `PYRE_FOR_ITER_GATE_DIAG`,
`PYRE_GC_DIAG`, `PYRE_GC_FREELIST_DIAG`, `PYRE_GEN_ENTRY_DIAG`,
`PYRE_JD1_DEBUG`, `PYRE_JD1_DUMP`,
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
`PYRE_UNJOURNALED_SITE`,
`PYRE_VSTACK_EXACT_AUDIT`, `PYRE_VSTACK_KEEP_REORDER`, `PYRE_VSTACK_NO_EXACT`,
`PYRE_WASM_DUMP_BAD_TRACE`, `PYRE_WASM_EXEC_TRACE`, `PYRE_WASM_FBW_CENSUS`,
`PYRE_WASM_GUARD_CENSUS`, `PYRE_WASM_JIT_STATS`, `PYRE_WASM_CALL_HIST`,
`PYRE_WASM_NO_CACHE`, `PYRE_WASM_STARTUP_TRACE`,
`PYRE_WASM_TRACE_ENTRY_CENSUS`.

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

`PYRE_CHECK_INHERIT_ENV` is the other odd one: an A/B switch, not a report.
`check.py` starts a benchmark child from an allowlisted environment because the
inherited one is startup allocation and moves `guard_failures`; setting the
gate restores the whole-environment copy, so the size of that effect stays
measurable on one binary. It goes when the allowlist stops being the thing
under measurement.

## §7 — General MAJIT gates

These translator and metainterpreter controls are inert unless explicitly set,
except for the function-pointer lowering switch, which is a build configuration
input.

| gate | default polarity | what it gates / retirement condition |
|---|---|---|
| `PYRE_CALLEE_CENSUS` | OFF | build-time census of resolved and unresolved callees; retire when all supported callees are classified without this diagnostic |
| `PYRE_CALLEE_CENSUS_ROWS` | OFF | row cap for that census; retire with `PYRE_CALLEE_CENSUS` |
| `PYRE_DESCR_POOL_CENSUS` | OFF | reports descriptor interning and duplication; retire when descriptor identity is covered by ordinary tests |
| `PYRE_FNPTR_INDIRECT` | OFF | enables indirect function-pointer lowering; retained as a build configuration switch |
| `PYRE_PROBE14` | OFF | reports discarded reference-constant relocations; retire when relocation preservation is covered by ordinary tests |
| `PYRE_VABLE_IDX_PROBE` | OFF | reports whether virtualizable array indices are constant; retire when all supported index shapes are covered by ordinary tests |

## Summary

| bucket | count |
|---|---|
| not gates (identifiers) | 12 |
| live default-ON, kept until epic closes | 7 |
| diagnostics (OFF) | ~35 |
| default-OFF experiments | 2 |
| config / value / master | ~17 |
| test harness | 1 |
