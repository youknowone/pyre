# majit Compatibility Matrix

Last updated: March 13, 2026

This file is the Phase 0 seed artifact for tracking equivalence with the in-tree RPython JIT sources.

## Legend

- `implemented`: intended to work and covered by focused tests
- `partial`: some behavior exists, but parity is incomplete
- `stubbed`: code path exists, but semantics are placeholder or incomplete
- `unsupported`: not implemented and should fail loudly if reached

## Subsystem Status

| Subsystem | RPython Reference | majit Status | Notes |
|---|---|---|---|
| IR opcode surface | `rpython/jit/metainterp/resoperation.py` | `partial` | `majit-ir` defines a broad opcode surface, but backend support is not complete. |
| Trace recorder | `rpython/jit/metainterp/pyjitpl.py` | `partial` | Manual trace building works for the example interpreters; snapshot/resume parity is missing, but the arithmetic/guard runtime seam now has normalized parity helpers plus dedicated `jit_interp` integration harnesses for inline arithmetic, simple booleanization, internal `if/else` guard emission, register-valued branch expressions, config-aware residual calls, helper-call lowering to `CallN` / `CallI` / `CallPureI` through explicit, bare inferred, and `auto_calls = true` paths, macro-generated `#[jit_inline]` helper lowering to inlined arithmetic sub-`JitCode`, nested inline helpers plus nested inline `CallPureI` / `CallN` with local helper-policy inference, tested integer-helper arity `0..=8`, and branch-group guard/close-loop behavior, and the generic `JitCode` runtime now executes with concrete int shadow state plus a narrow raw `MIFrame` inline-call subset for int argument/return register propagation instead of a direct per-arm trace loop. |
| Warm state / jitcell lifecycle | `rpython/jit/metainterp/warmstate.py` | `partial` | Hotness tracking exists, but full procedure-token ownership and driver integration are incomplete. |
| Resume / blackhole | `rpython/jit/metainterp/resume.py` | `partial` | `EncodedResumeData` now has full `encode()`/`decode()` roundtrip with RPython-style tagged numbering (TAG_INT, TAG_CONST, TAG_FAILARG, TAG_VIRTUAL plus sentinel slots), sparse fail-args are compact-numbered with raw-slot remapping, `ResumeDataLoopMemo` shares constant pools across guards, pending field writes roundtrip through the same encoded snapshot, all 5 virtual kinds encode/decode correctly, virtual materialization works, blackhole now has a `BlackholeMemory` trait for pluggable memory access (gc_load/gc_store/call dispatch) beyond the default placeholder, and `run_with_blackhole_fallback()` can replay guard failures from compiled loops or bridges. General jitframe reconstruction from arbitrary stack frames is still missing. |
| Optimizer pipeline | `rpython/jit/metainterp/optimizeopt/` | `partial` | 8-pass default pipeline (IntBounds, Rewrite, Virtualize, String, Pure, Guard, Simplify, Heap) plus loop peeling (unroll), short preamble extraction/instantiation, bridgeopt knowledge propagation, earlyforce virtual ref resolution, virtualstate export/import, and vector auto-vectorization (dependency graph + cost model + SIMD pack groups). |
| Cranelift backend | `rpython/jit/backend/*` | `partial` | Integer-heavy traces are exercised; mixed-type exits preserve `Int`/`Ref`/`Float`, generic `GC_LOAD*` / `GC_STORE*`, raw load/store, interior-field ops, and rewrite-emitted `ZERO_ARRAY` now execute for the supported subset, rewrite-generated GC ops execute, supported `CALL_*` / `COND_CALL_*` paths register shadow roots around collecting calls, returned dead-frame refs stay rooted until the frame is dropped, exact-match `GUARD_EXCEPTION`, `GUARD_NO_EXCEPTION`, `SAVE_EXC_CLASS`, `SAVE_EXCEPTION`, and `RESTORE_EXCEPTION` now execute for the supported subset with `Backend::grab_exc_class()` / `Backend::grab_exc_value()`-visible dead-frame exception state, attached bridges can now run with a still-pending exception so bridge-entry exception ops can consume it, high-level GC traces are normalized and rewritten automatically when a GC runtime is configured, descriptorless string/unicode `NEW*` / len / get / set / copy / hash ops execute with a builtin cached-hash + length + items layout, `CALL_ASSEMBLER_*` can dispatch to already-compiled finish-only callee loops with redirect support, `FORCE_TOKEN` plus `GUARD_NOT_FORCED_2` can capture a post-finish force snapshot that `Backend::force()` materializes later, and `CALL_MAY_FORCE_*` plus the immediately following `GUARD_NOT_FORCED` now execute for a narrow forced-call subset with typed preview deadframes, savedata, and GC-stressed ref-result coverage. |
| GC rewriter | `rpython/jit/backend/llsupport/rewrite.py` | `partial` | Nursery/write-barrier rewriting exists, explicit result numbering is now preserved for traced values, and the Cranelift backend consumes rewritten output automatically for the supported subset. |
| GC runtime | `rpython/memory/gc/incminimark.py` | `partial` | Basic nursery + old-gen collector exists; compiled allocation helpers, supported indirect calls, and dead-frame refs can register shadow roots across nursery collection; `SafepointMap` + `CompiledCodeRegistry` provide generic stack-map scanning infrastructure for compiled code regions with bitmap-based GC ref enumeration. Card marking and incremental semantics are still incomplete. |
| Driver macros/runtime | `rpython/rlib/jit.py` | `stubbed` | Current macros annotate metadata only; they do not yet drive tracing lifecycle like `JitDriver`. |

## Backend Policy

The Cranelift backend must obey the following rules:

1. Unsupported opcodes must return `BackendError::Unsupported`.
2. Void opcodes are not allowed to silently degrade into no-ops unless they are explicitly documented as semantic no-ops.
3. Guard exits and `FINISH` exits must preserve `Int`, `Ref`, and `Float` values exactly.

## Immediate Gaps

These are the highest-priority items currently blocking equivalence:

- Guard-bearing `CALL_ASSEMBLER_*` and recursive call dispatch are incomplete (only finish-only callee loops work).
- Only the narrow single-call + immediate-guard `CALL_MAY_FORCE_*` / `GUARD_NOT_FORCED` subset is supported; general savedata/resume paths and broader jitframe state are still missing.
- Full jitframe reconstruction from arbitrary caller stack frames is not yet implemented; `EncodedResumeData` now roundtrips compact fail-arg numbering plus pending writes, but the backend does not yet emit complete frame layout metadata for all guard sites.
- Card marking and incremental GC semantics are incomplete.
- Driver macros do not yet implement RPython-equivalent `jit_merge_point` / `can_enter_jit` tracing lifecycle (currently annotation-only).
- `#[jit_inline]` exists only for simple integer helpers; bare `calls = { helper }` entries can now infer helper policy from `#[jit_inline]` / `#[elidable]` / `#[dont_look_inside]`, `#[jit_interp(auto_calls = true)]` can now infer direct top-level helper calls without a call map, and nested `#[jit_inline]` helpers can now infer direct helper calls without a local map. The current integer-helper call subset is tested through arity `0..=8`, but fully automatic whole-module helper discovery, richer helper signatures, and helper calls beyond that current integer subset are still missing.
- Virtual ref finishing and full FORCE_TOKEN/GUARD_NOT_FORCED lifecycle need completion.

## Next Expansion

The next matrix revision should split backend opcode coverage into:

- control flow and guards
- integer arithmetic and comparisons
- float and ref value handling
- GC-sensitive loads/stores/allocation
- calls and may-force semantics
- strings/unicode
- bridges / call-assembler / virtual refs
