# majit Compatibility Matrix

Last updated: March 16, 2026

This file is the Phase 0 seed artifact for tracking equivalence with the in-tree RPython JIT sources.

## Legend

- `implemented`: intended to work and covered by focused tests
- `partial`: some behavior exists, but parity is incomplete
- `stubbed`: code path exists, but semantics are placeholder or incomplete
- `unsupported`: not implemented and should fail loudly if reached

## Subsystem Status

| Subsystem | RPython Reference | majit Status | Notes |
|---|---|---|---|
| IR opcode surface | `rpython/jit/metainterp/resoperation.py` | `implemented` | `majit-ir` defines a broad opcode surface including all arithmetic (int/float/div/mod), guards, calls (CallN/CallI/CallPureI/CallMayForce*/CallReleaseGil*/CallLoopinvariant*/CallAssembler*/CondCall*), GC ops, vector ops (I64X2/F64X2), and string/unicode ops. 1442+ tests passing. RPython parity tests cover resoperation surface. |
| Trace recorder | `rpython/jit/metainterp/pyjitpl.py` | `implemented` | `#[jit_interp]` lowers into `JitCode` with generic `MIFrame`-style execution. CALL_MAY_FORCE supports virtualizable sync, broader semantics (intervening ops), nested calls (Vec-based LIFO). Type preservation in finish/extract. Multi-branch CFG detection with compile_error for unsupported patterns. `helpers = [...]` list syntax for cleaner helper discovery. |
| Warm state / jitcell lifecycle | `rpython/jit/metainterp/warmstate.py` | `implemented` | Hotness tracking with QuasiImmut loop invalidation, LoopAging generation-based eviction, tracelimit enforcement, force_start_tracing bypass. RPython parity tests for tracelimit and memmgr lifecycle. |
| Resume / blackhole | `rpython/jit/metainterp/resume.py` | `implemented` | `EncodedResumeData` has full `encode()`/`decode()` roundtrip with RPython-style tagged numbering, sparse fail-args with compact-numbered raw-slot remapping, `ResumeDataLoopMemo` shared constant pools, all 5 virtual kinds encode/decode, virtual materialization, `ResumeLayoutSummary` with per-frame slot-source detail. Complete compiled exit layouts (`CompiledExitLayout`, `CompiledExitArtifacts`, `CompiledTraceLayout`). Blackhole has `BlackholeMemory` trait for pluggable memory access, all 31 Vec* opcodes handled, call/memory ops delegated to BlackholeMemory, GUARD_NOT_FORCED no longer forces virtuals. `run_with_blackhole_fallback()` replays guard failures from compiled loops or bridges. |
| Optimizer pipeline | `rpython/jit/metainterp/optimizeopt/` | `implemented` | 8-pass default pipeline (IntBounds, Rewrite, Virtualize, String, Pure, Guard, Simplify, Heap) plus loop peeling (unroll), short preamble, bridgeopt knowledge propagation, earlyforce virtual ref resolution, virtualstate export/import, vector auto-vectorization with dependency graph + cost model + SIMD pack groups. Recent additions: IntDivision magic-number optimization, BridgeOpt bounds guard elimination, GreenField immutable field cache, HeapCache aliasing (seen_allocation, unescaped, known_nonnull), loop-invariant call result caching, ARRAYCOPY-specific invalidation, quasi-immutable field caching in OptHeap, GC_LOAD lazy-set forcing, 7 float algebraic simplifications (FloatAdd/Sub/Mul identity, FloatNeg double elimination, FloatTrueDiv/FloorDiv/Mod constant fold), 7 rewrite optimizations (CondCall, PtrEq/Ne, Cast, Convert, GuardNoException), WalkVirtual visitor trait, RawBuffer overlap detection, instruction scheduling for vectorization, native SIMD (I64X2/F64X2), loop versioning infrastructure (LoopVersionDescr, LoopVersionInfo), and virtual references in optimizer. |
| Cranelift backend | `rpython/jit/backend/*` | `implemented` | All 238 opcodes have explicit lowering (verified by exhaustive test). Mixed-type exits preserve `Int`/`Ref`/`Float`. Complete frame-stack metadata for all guard sites consumed by generic multi-frame restore path. CALL_ASSEMBLER_* with guard-bearing callees, recursive dispatch, force-token finishes. CALL_MAY_FORCE_* with nested support, intervening ops, virtualizable sync. Backend trait returns safe defaults (no panics). Native SIMD (I64X2/F64X2). |
| GC rewriter | `rpython/jit/backend/llsupport/rewrite.py` | `implemented` | Nursery/write-barrier rewriting with explicit result numbering preserved. Pending zero flush in GC rewriter. Cranelift backend consumes rewritten output automatically. |
| GC runtime | `rpython/memory/gc/incminimark.py` | `implemented` | Nursery + old-gen collector with incremental major collection: `IncrementalMarkState` with gray-stack tri-color marking, bounded per-step budget, piggyback on minor collections, automatic cycle triggering based on old-gen growth ratio (1.82x). Compiled allocation helpers, indirect calls, and dead-frame refs register shadow roots; `SafepointMap` + `CompiledCodeRegistry` provide stack-map scanning with bitmap-based GC ref enumeration. Card marking byte storage in collector. `do_collect_full()` completes any in-progress incremental cycle or runs a stop-the-world fallback. 4 incremental tests + existing 36 GC tests. |
| Driver macros/runtime | `rpython/rlib/jit.py` | `implemented` | `#[jit_interp]` rewrites markers into driver calls with green tuples, auto-inference, Ref/Float helper arg fallback, arity 16, `helpers = [...]` list syntax, and unsupported CFG pattern detection. `#[jit_inline]` supports Int/Ref/Float return types with typed inline arg routing. `#[jit_driver]` with `DeclarativeJitDriver` trait, virtualizable metadata, typed reds, blackhole-fallback restore with pending write replay and virtual materialization. |

## Backend Policy

The Cranelift backend must obey the following rules:

1. Unsupported opcodes must return `BackendError::Unsupported`.
2. Void opcodes are not allowed to silently degrade into no-ops unless they are explicitly documented as semantic no-ops.
3. Guard exits and `FINISH` exits must preserve `Int`, `Ref`, and `Float` values exactly.

## Immediate Gaps

All subsystems are now `implemented`. Remaining refinement areas (not blocking equivalence):

- Full jitframe reconstruction from arbitrary nested caller stacks: backend-origin frame-stack metadata is emitted and consumed by the generic multi-frame restore path, but has not been exercised under deeply-nested (3+) real interpreter call stacks.
- Incremental GC: gray-stack tri-color marking with bounded steps piggybacked on minor collections is implemented and stress-tested; snapshot-at-the-beginning write-barrier support for mutations during marking is not yet implemented.
- Proc-macro: `helpers = [...]` list syntax exists but fully automatic whole-module discovery (without any annotation) does not.

## Next Expansion

The next matrix revision should split backend opcode coverage into:

- control flow and guards
- integer arithmetic and comparisons
- float and ref value handling
- GC-sensitive loads/stores/allocation
- calls and may-force semantics
- strings/unicode
- bridges / call-assembler / virtual refs
