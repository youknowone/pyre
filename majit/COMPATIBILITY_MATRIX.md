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
| Trace recorder | `rpython/jit/metainterp/pyjitpl.py` | `partial` | Manual trace building works for all 8 example interpreters; `#[jit_interp]` lowers into `JitCode` with generic `MIFrame`-style execution. Snapshot/resume parity exists for the supported subset. CALL_MAY_FORCE now supports virtualizable sync (before/after residual call), broader semantics allowing intervening ops before GUARD_NOT_FORCED, and `on_compile_error` callback. Type preservation in `finish_and_compile` and `extract_live_values`. Full multi-branch CFG and automatic whole-module call discovery still missing. |
| Warm state / jitcell lifecycle | `rpython/jit/metainterp/warmstate.py` | `partial` | Hotness tracking exists with QuasiImmut loop invalidation notifier, but full procedure-token ownership and driver integration are incomplete. |
| Resume / blackhole | `rpython/jit/metainterp/resume.py` | `implemented` | `EncodedResumeData` has full `encode()`/`decode()` roundtrip with RPython-style tagged numbering, sparse fail-args with compact-numbered raw-slot remapping, `ResumeDataLoopMemo` shared constant pools, all 5 virtual kinds encode/decode, virtual materialization, `ResumeLayoutSummary` with per-frame slot-source detail. Complete compiled exit layouts (`CompiledExitLayout`, `CompiledExitArtifacts`, `CompiledTraceLayout`). Blackhole has `BlackholeMemory` trait for pluggable memory access, all 31 Vec* opcodes handled, call/memory ops delegated to BlackholeMemory, GUARD_NOT_FORCED no longer forces virtuals. `run_with_blackhole_fallback()` replays guard failures from compiled loops or bridges. |
| Optimizer pipeline | `rpython/jit/metainterp/optimizeopt/` | `implemented` | 8-pass default pipeline (IntBounds, Rewrite, Virtualize, String, Pure, Guard, Simplify, Heap) plus loop peeling (unroll), short preamble, bridgeopt knowledge propagation, earlyforce virtual ref resolution, virtualstate export/import, vector auto-vectorization with dependency graph + cost model + SIMD pack groups. Recent additions: IntDivision magic-number optimization, BridgeOpt bounds guard elimination, GreenField immutable field cache, HeapCache aliasing (seen_allocation, unescaped, known_nonnull), loop-invariant call result caching, ARRAYCOPY-specific invalidation, quasi-immutable field caching in OptHeap, GC_LOAD lazy-set forcing, 7 float algebraic simplifications (FloatAdd/Sub/Mul identity, FloatNeg double elimination, FloatTrueDiv/FloorDiv/Mod constant fold), 7 rewrite optimizations (CondCall, PtrEq/Ne, Cast, Convert, GuardNoException), WalkVirtual visitor trait, RawBuffer overlap detection, instruction scheduling for vectorization, native SIMD (I64X2/F64X2), loop versioning infrastructure (LoopVersionDescr, LoopVersionInfo), and virtual references in optimizer. |
| Cranelift backend | `rpython/jit/backend/*` | `partial` | Integer and float traces execute; mixed-type exits preserve `Int`/`Ref`/`Float`, GC_LOAD*/GC_STORE*, raw load/store, interior-field ops, ZERO_ARRAY, GC ops, CALL_*/COND_CALL_* with shadow roots, exception handling (GUARD_EXCEPTION, GUARD_NO_EXCEPTION, SAVE_EXC_CLASS/SAVE_EXCEPTION/RESTORE_EXCEPTION), string/unicode NEW*/len/get/set/copy/hash, CALL_ASSEMBLER_* dispatch with redirect support. CALL_MAY_FORCE_* with broader semantics (intervening ops allowed before GUARD_NOT_FORCED), guard-bearing callees supported in CallAssembler, force-token finish shapes relaxed. Backend trait panics converted to safe defaults (BackendError). Complete frame-stack metadata for all guard sites with `compiled_guard_frame_stacks()` query API. |
| GC rewriter | `rpython/jit/backend/llsupport/rewrite.py` | `implemented` | Nursery/write-barrier rewriting with explicit result numbering preserved. Pending zero flush in GC rewriter. Cranelift backend consumes rewritten output automatically. |
| GC runtime | `rpython/memory/gc/incminimark.py` | `partial` | Basic nursery + old-gen collector; compiled allocation helpers, indirect calls, and dead-frame refs register shadow roots; `SafepointMap` + `CompiledCodeRegistry` provide stack-map scanning with bitmap-based GC ref enumeration (with safepoint/registry tests). Card marking byte storage in collector (4 tests). Full incremental GC semantics are still incomplete. |
| Driver macros/runtime | `rpython/rlib/jit.py` | `partial` | `#[jit_interp]` rewrites `jit_merge_point!` / `can_enter_jit!` into driver calls for both legacy and explicit bindings, with optional green tuples, auto-inference for `CallMayForce*`/`CallReleaseGil*`/`CallLoopinvariant*`, Ref/Float helper arg fallback, and host call arity extended to 16. `#[jit_driver]` validates green/red overlap plus virtualizable metadata with `DeclarativeJitDriver` trait. Runtime drivers validate type mismatches, preserve typed `Ref`/`Float` reds, expose typed raw fast paths, and route through `extract_live_values()` / `restore_values()`. Three virtualizable layers with opt-in automatic heap-sync. Narrow blackhole-fallback restore path with typed resume-frame projection, pending write replay, nested virtual graphs, and cached virtual materialization. `call_assembler_int_by_number_typed` for typed dispatch. Full declarative lifecycle parity is still incomplete. |

## Backend Policy

The Cranelift backend must obey the following rules:

1. Unsupported opcodes must return `BackendError::Unsupported`.
2. Void opcodes are not allowed to silently degrade into no-ops unless they are explicitly documented as semantic no-ops.
3. Guard exits and `FINISH` exits must preserve `Int`, `Ref`, and `Float` values exactly.

## Immediate Gaps

These are the highest-priority items currently blocking equivalence:

- Full jitframe reconstruction from arbitrary caller stack frames is not yet implemented; `EncodedResumeData` roundtrips, compiled traces preserve terminal exit layouts, and complete frame-stack metadata is now available for all guard sites, but the backend does not yet perform full caller-frame jitframe layout reconstruction for arbitrary nested frames.
- Full incremental GC semantics are incomplete; card marking byte storage and basic card-marking tests exist, but full incremental write-barrier and remembered-set semantics are still partial.
- Driver macros/runtime still lack full RPython-equivalent declarative lifecycle coverage; the supported subset is broad (see table above), but `#[jit_inline]` still only covers simple integer helpers, ref/float helper return signatures are unsupported, and fully automatic whole-module helper discovery does not exist.
- Multi-branch control-flow graphs beyond simple `if/else` do not exist in the proc-macro lowering seam.

## Next Expansion

The next matrix revision should split backend opcode coverage into:

- control flow and guards
- integer arithmetic and comparisons
- float and ref value handling
- GC-sensitive loads/stores/allocation
- calls and may-force semantics
- strings/unicode
- bridges / call-assembler / virtual refs
