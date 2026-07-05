# PYRE_* gate triage

**Status**: living record. WS4 deliverable of `rework.md` (finding F5, gate
debt). Audited 2026-07-05 on branch `pc-map` by reading the actual
read-expression at every source site (four census passes). Polarity rule:
`is_some()`/`=="1"`/`.unwrap_or(false)` → default **OFF**; `is_none()` /
`!= Ok("0")` / `.unwrap_or(true)` → default **ON**; a parse of
number/path/list/mode → **VALUE** (config, not a boolean gate).

The charter (§3.6, A7) says a gate is a staging area, not a home: each live
default-ON experiment gate is kept only until its epic closes, then its OFF
path is deleted and the gate retired. This table is the standing list of what
to retire and when.

## Headline

The raw `rg 'PYRE_[A-Z0-9_]+'` count (~119) overstates the debt. **~20 of the
matches are not env gates at all** (Rust consts, macro-generated identifiers,
runtime symbols, or comment-only dead references). Real live env vars ≈ 99, of
which ~33 are default-ON experiment gates. Of those 33, the **wasm trio** and
the **#171 pair** were cleanly settled (epic closed / merged) and are retired
this pass; the rest are load-bearing kill switches for open reworks.

## §1 — Retired this pass (5)

Hardwired ON. Behaviour is byte-identical (each was default-ON already; only
the opt-out capability is gone). The **wasm trio** removed the env read +
guest export + `set_*` + `AtomicBool` static, reader fns collapsed to `true`;
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

## §3 — Dead (9): no env read site

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

## §4 — Live default-ON gates KEPT (retire when the epic closes)

Each is default-ON but still a load-bearing kill switch for an open rework; its
OFF path is a needed safety net. Retire at the listed trigger (A7).

| var | subsystem | retire when |
|---|---|---|
| PYRE_FULL_BODY_WALK | master: full-body-walk tracer vs trait leg | WS1 / #344 / #366 single-executor lands (the OFF path *is* the trait leg #344 deletes) |
| PYRE_FBW_INLINE, _INLINE_MULTIFRAME, _INLINE_NSFOLD, _LOOP_CALLEE_CA | walker inlining (#62/#68/gap-10) | same epic cluster |
| PYRE_FBW_CALL_ASSEMBLER, _NO_REPLAY_EXIT, _RAISE, _REC_CA | walker return/raise/recursion | same |
| PYRE_FBW_ABORT_FLUSH, _BRANCH_FLUSH, _END_FLUSH, _BRIDGE_LOCAL_SEED | shadow-stack flush/seed on resume | same (couples to F1 resume convergence) |
| PYRE_FBW_BUILTIN_FOLD, _LOADGLOBAL_FOLD, _LOADNAME_FOLD, _STORENAME_FOLD | const-folds in walker bodies | same (fold correctness interlocks with the walker) |
| PYRE_FBW_NESTED_RESID_ABORT | nested-residual abort vs replay | same |
| PYRE_TWO_PHASE_RTYPE, PYRE_TUPLE_PER_SHAPE_CLASSDEF | rtyper prepass / per-shape tuple classdef | WS2 / #346 rtyper epic |
| PYRE_ORIGINAL_BOXES | greens++reds original_boxes index shape | box-identity #202 / resume F1 |
| PYRE_MIR_FRAMESTATE | framestate-threaded MIR lowering | MIR front-end #176/#181/#346 |
| PYRE_GC_ITEMSBLOCK, PYRE_GC_PREBUILT_REMEMBER, PYRE_GC_INTERP_COLLECT | GC-managed items / prebuilt minor-skip / interp collect A/B | WS3 / #355 / F3 GC rework |
| PYRE_57_INLINE_NEXT | FOR_ITER inline-next fastpath | #57 / task#50 (has admitted unsoundness) |
| PYRE_CL_NO_CLOSING_JUMP | cranelift attached-loop closing jump | #245 cranelift perf (explicit rollback hatch) |
| PYRE_NO_INNER_CLOSE | cross-loop-cut inner-close override | #152 cross-loop residency |

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
- **Default-OFF experiments (~14)** — not "already activated", so out of scope:
  `PYRE_FBW_VABLE_SCALAR_CA`, `PYRE_SINGLE_PASS`, `PYRE_KEPT_OVERRIDE`,
  `PYRE_P2_AUTHORITATIVE`, `_COMPILE`, `_DRAIN`, `_FRAMESTACK`, `_FS_COMPILE`,
  `PYRE_AUTHORITATIVE`, `PYRE_STRICT_TARGET_TO_PATH`, `PYRE_SAME_GREENKEY`,
  `PYRE_INNER_CLOSE`, `PYRE_RELAX_124`, `PYRE_NO_DE`.
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
| retired this pass | 5 |
| not gates (identifiers) | 11 |
| dead (no read site) | 9 |
| live default-ON, kept until epic closes | ~28 |
| diagnostics (OFF) | ~34 |
| default-OFF experiments | ~14 |
| config / value / master | ~18 |
| test harness | 1 |
