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
| Trace recorder | `rpython/jit/metainterp/pyjitpl.py` | `partial` | `#[jit_interp]` lowers into `JitCode` with generic `MIFrame`-style execution. Multi-branch CFG detected but not lowered (compile_error). `helpers = [...]` list syntax available but fully automatic whole-module discovery without annotation does not exist. RPython's codewriter handles arbitrary control flow and general interpreter shapes; majit covers sequential/binary-branch patterns. |
| Warm state / jitcell lifecycle | `rpython/jit/metainterp/warmstate.py` | `partial` | Hotness tracking, QuasiImmut, LoopAging, tracelimit work. Full procedure-token ownership and jitcell lifecycle matching RPython's warmstate.py is not complete — jitcell states, token invalidation chains, and multi-entry-point ownership are not fully replicated. |
| Resume / blackhole | `rpython/jit/metainterp/resume.py` | `implemented` | `EncodedResumeData` has full `encode()`/`decode()` roundtrip with RPython-style tagged numbering, sparse fail-args with compact-numbered raw-slot remapping, `ResumeDataLoopMemo` shared constant pools, all 5 virtual kinds encode/decode, virtual materialization, `ResumeLayoutSummary` with per-frame slot-source detail. Complete compiled exit layouts (`CompiledExitLayout`, `CompiledExitArtifacts`, `CompiledTraceLayout`). Blackhole has `BlackholeMemory` trait for pluggable memory access, all 31 Vec* opcodes handled, call/memory ops delegated to BlackholeMemory, GUARD_NOT_FORCED no longer forces virtuals. `run_with_blackhole_fallback()` replays guard failures from compiled loops or bridges. |
| Optimizer pipeline | `rpython/jit/metainterp/optimizeopt/` | `implemented` | 8-pass default pipeline (IntBounds, Rewrite, Virtualize, String, Pure, Guard, Simplify, Heap) plus loop peeling (unroll), short preamble, bridgeopt knowledge propagation, earlyforce virtual ref resolution, virtualstate export/import, vector auto-vectorization with dependency graph + cost model + SIMD pack groups. Recent additions: IntDivision magic-number optimization, BridgeOpt bounds guard elimination, GreenField immutable field cache, HeapCache aliasing (seen_allocation, unescaped, known_nonnull), loop-invariant call result caching, ARRAYCOPY-specific invalidation, quasi-immutable field caching in OptHeap, GC_LOAD lazy-set forcing, 7 float algebraic simplifications (FloatAdd/Sub/Mul identity, FloatNeg double elimination, FloatTrueDiv/FloorDiv/Mod constant fold), 7 rewrite optimizations (CondCall, PtrEq/Ne, Cast, Convert, GuardNoException), WalkVirtual visitor trait, RawBuffer overlap detection, instruction scheduling for vectorization, native SIMD (I64X2/F64X2), loop versioning infrastructure (LoopVersionDescr, LoopVersionInfo), and virtual references in optimizer. |
| Cranelift backend | `rpython/jit/backend/*` | `partial` | All 238 opcodes have explicit lowering. Frame-stack metadata emitted for all guard sites and consumed by generic multi-frame restore path, but arbitrary nested caller-frame reconstruction (RPython resume.py level) is not fully exercised. CALL_ASSEMBLER/CALL_MAY_FORCE broadened but not fully general. |
| GC rewriter | `rpython/jit/backend/llsupport/rewrite.py` | `implemented` | Nursery/write-barrier rewriting with explicit result numbering preserved. Pending zero flush in GC rewriter. Cranelift backend consumes rewritten output automatically. |
| GC runtime | `rpython/memory/gc/incminimark.py` | `partial` | Nursery + oldgen + incremental marking + card marking + pending zero flush. Safepoint infrastructure exists but generic stack-map scanning under real compiled-code GC stress is not yet claim-gated. RPython's incminimark has deeper integration with the JIT (jit_free/jit_pinning/jit_gc_step) that is not replicated. |
| Driver macros/runtime | `rpython/rlib/jit.py` | `partial` | `#[jit_interp]` covers sequential/binary-branch interpreter shapes with `helpers = [...]` and auto-inference. `#[jit_inline]` supports Int/Ref/Float returns. `#[jit_driver]` with `DeclarativeJitDriver`. Full RPython `JitDriver` lifecycle (multi-entry, procedure-entry, set_param, get_stats callback integration) is not fully replicated. Whole-module helper discovery without annotation does not exist. |

## Backend Policy

The Cranelift backend must obey the following rules:

1. Unsupported opcodes must return `BackendError::Unsupported`.
2. Void opcodes are not allowed to silently degrade into no-ops unless they are explicitly documented as semantic no-ops.
3. Guard exits and `FINISH` exits must preserve `Int`, `Ref`, and `Float` values exactly.

## Immediate Gaps

5 subsystems are `implemented`, 4 remain `partial`. The `partial` items share a common theme: the infrastructure exists and tests pass, but RPython-level generality (arbitrary interpreter shapes, deep caller stacks, full GC-JIT integration, complete driver lifecycle) is not yet claim-gated.

Remaining gaps in priority order:

1. **Cranelift backend**: arbitrary nested caller-frame jitframe reconstruction via backend-origin metadata is not fully exercised.
2. **Trace recorder**: multi-branch CFG lowering and whole-module helper discovery.
3. **GC runtime**: generic stack-map scanning under real compiled-code GC stress; deeper JIT-GC integration (jit_free, jit_pinning).
4. **Warm state**: full procedure-token ownership, jitcell state machine, multi-entry-point management.
5. **Driver macros**: full RPython `JitDriver` lifecycle (set_param, get_stats, multi-entry).

## Next Expansion

The next matrix revision should split backend opcode coverage into:

- control flow and guards
- integer arithmetic and comparisons
- float and ref value handling
- GC-sensitive loads/stores/allocation
- calls and may-force semantics
- strings/unicode
- bridges / call-assembler / virtual refs
