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
| Static analyzer / translator | `rpython/jit/codewriter/*` + translation-time front-end | `implemented` | `majit-analyze` parses multiple Rust source files, extracts opcode dispatch arms, resolves cross-file trait impls and helper classifications, collects type layouts, and generates tracing helper code. Consumed by `pyre-mjit` build.rs. Classifies 39/40 pyre opcode arms (arithmetic, comparisons, local/const access, jumps, returns, namespace ops, iterator cleanup, stack manipulation, function calls, noop). Complex CFG (loop/while/for in match arms) generates abort JitCode for interpreter fallback matching RPython dont_look_inside semantics. Only the wildcard `_` arm remains unclassified. |
| Trace recorder | `rpython/jit/metainterp/pyjitpl.py` | `implemented` | `#[jit_interp]` lowers into `JitCode` with generic `MIFrame`-style execution. Multi-branch match expressions lowered to IntEq + branch_reg_zero guard chains. `#[jit_module]` for auto-discovery. `helpers = [...]` list syntax. Complex CFG (loop/while/for in arms) generates abort JitCode for interpreter fallback instead of compile_error. |
| Warm state / jitcell lifecycle | `rpython/jit/metainterp/warmstate.py` | `implemented` | JitCell state machine (NotHot/Tracing/Compiled/Invalidated/DontTraceHere) with procedure-token ownership, `set_param()` runtime tuning (threshold/trace_limit/bridge_threshold/function_threshold/max_inline_depth), `get_stats()` snapshots, `gc_cells()` dead cell cleanup, LoopAging eviction, QuasiImmut invalidation, tracelimit enforcement. 9 parity tests. |
| Resume / blackhole | `rpython/jit/metainterp/resume.py` | `implemented` | `EncodedResumeData` has full `encode()`/`decode()` roundtrip with RPython-style tagged numbering, sparse fail-args with compact-numbered raw-slot remapping, `ResumeDataLoopMemo` shared constant pools, all 5 virtual kinds encode/decode, virtual materialization, `ResumeLayoutSummary` with per-frame slot-source detail. Complete compiled exit layouts (`CompiledExitLayout`, `CompiledExitArtifacts`, `CompiledTraceLayout`). Blackhole has `BlackholeMemory` trait for pluggable memory access, all 31 Vec* opcodes handled, call/memory ops delegated to BlackholeMemory, GUARD_NOT_FORCED no longer forces virtuals. `run_with_blackhole_fallback()` replays guard failures from compiled loops or bridges. |
| Optimizer pipeline | `rpython/jit/metainterp/optimizeopt/` | `implemented` | 8-pass default pipeline (IntBounds, Rewrite, Virtualize, String, Pure, Guard, Simplify, Heap) plus loop peeling (unroll), short preamble, bridgeopt knowledge propagation, earlyforce virtual ref resolution, virtualstate export/import, vector auto-vectorization with dependency graph + cost model + SIMD pack groups. Recent additions: IntDivision magic-number optimization, BridgeOpt bounds guard elimination, GreenField immutable field cache, HeapCache aliasing (seen_allocation, unescaped, known_nonnull), loop-invariant call result caching, ARRAYCOPY-specific invalidation, quasi-immutable field caching in OptHeap, GC_LOAD lazy-set forcing, 7 float algebraic simplifications (FloatAdd/Sub/Mul identity, FloatNeg double elimination, FloatTrueDiv/FloorDiv/Mod constant fold), 7 rewrite optimizations (CondCall, PtrEq/Ne, Cast, Convert, GuardNoException), WalkVirtual visitor trait, RawBuffer overlap detection, instruction scheduling for vectorization, native SIMD (I64X2/F64X2), loop versioning infrastructure (LoopVersionDescr, LoopVersionInfo), and virtual references in optimizer. |
| Cranelift backend | `rpython/jit/backend/*` | `implemented` | All 238 opcodes have explicit lowering (exhaustive test). Frame-stack metadata emitted for all guard sites and exercised through E2E compiled code execution: guard failure metadata, multi-guard query, bridge frame_stack, call_assembler callee guard, mixed-type slot_types. CALL_ASSEMBLER with guard-bearing callees + recursive dispatch. CALL_MAY_FORCE with nested + intervening ops + virtualizable sync. Native SIMD (I64X2/F64X2). |
| GC rewriter | `rpython/jit/backend/llsupport/rewrite.py` | `implemented` | Nursery/write-barrier rewriting with explicit result numbering preserved. Pending zero flush in GC rewriter. Cranelift backend consumes rewritten output automatically. |
| GC runtime | `rpython/memory/gc/incminimark.py` | `implemented` | Nursery + oldgen + incremental marking + card marking + pending zero flush + JIT-GC integration hooks (`jit_remember_young_pointer`, `can_optimize_cond_call`, `gc_step`). Safepoint stress tests under allocation pressure, write-barrier-during-marking correctness, mutation reachability preservation. 93 GC tests. |
| Driver macros/runtime | `rpython/rlib/jit.py` | `implemented` | `#[jit_interp]` with `helpers = [...]`, auto-inference, unsupported CFG detection. `#[jit_inline]` with Int/Ref/Float typed returns. `#[jit_module]` for module-level auto-discovery. `#[jit_driver]` with `DeclarativeJitDriver`, `set_param()`/`get_stats()`, typed reds, blackhole-fallback restore. JitHookInterface with real-pipeline compile/bridge/error/guard events (4 tests). FFI forcing with CallMayForce+GuardNotForced and exception propagation (3 E2E tests). |

## Backend Policy

The Cranelift backend must obey the following rules:

1. Unsupported opcodes must return `BackendError::Unsupported`.
2. Void opcodes are not allowed to silently degrade into no-ops unless they are explicitly documented as semantic no-ops.
3. Guard exits and `FINISH` exits must preserve `Int`, `Ref`, and `Float` values exactly.

## Immediate Gaps

All 10 subsystems are `implemented`.

No structural gaps remain. Breadth refinement areas:

- **Static analyzer**: 39/40 pyre opcode arms classified with 18 first-class TracePattern variants. Only `get_iter` and `make_function` remain Residual (matching RPython codewriter — these are genuine residual calls). Complex CFG generates abort JitCode for interpreter fallback.
- **E2E validation**: arbitrary-depth multi-frame restore tested up to depth 50. Exercising under real compiled interpreter stacks (e.g., pyre recursive calls) would provide final confidence.

## Next Expansion

The next matrix revision should split backend opcode coverage into:

- control flow and guards
- integer arithmetic and comparisons
- float and ref value handling
- GC-sensitive loads/stores/allocation
- calls and may-force semantics
- strings/unicode
- bridges / call-assembler / virtual refs
