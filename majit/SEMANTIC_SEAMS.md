# majit Semantic Seams

This document defines how `majit` can keep its `proc-macro + runtime`
architecture and still claim parity with RPython in a defensible way.

## Rule

Parity is checked at stable semantic seams, not by forcing a 1:1 file layout
with RPython.

`rpython/jit/metainterp/pyjitpl.py` is a large, monolithic meta-interpreter.
`majit` deliberately splits the same responsibilities across proc macros,
runtime helpers, trace recording, optimization, and backend crates. That split
is acceptable only if the split points are explicit and testable.

## Primary Seams

### 0. Translation-Time Analysis Seam

Input:
- interpreter source files

Output:
- extracted opcode dispatch metadata, helper classifications, trait/type info, generated tracing helpers

Current majit owners:
- [majit-analyze/src/lib.rs](/Users/al03219714/Projects/pypy/majit/majit-analyze/src/lib.rs)
- [majit-analyze/src/codegen.rs](/Users/al03219714/Projects/pypy/majit/majit-analyze/src/codegen.rs)
- [pyre-mjit/build.rs](/Users/al03219714/Projects/pypy/pyre/pyre-mjit/build.rs)

Rule:
- parity is checked by extracted dispatch shape, resolved helper classifications, and generated helper code behavior
- unclassified patterns must stay explicit residual/unclassified; they must not silently become optimized tracing paths

### 1. Macro Lowering Seam

Input:
- interpreter source AST

Output:
- calls into a small runtime tracing surface

Current majit owners:
- [jit_interp/mod.rs](/Users/al03219714/Projects/pypy/majit/majit-macros/src/jit_interp/mod.rs)
- [codegen_trace.rs](/Users/al03219714/Projects/pypy/majit/majit-macros/src/jit_interp/codegen_trace.rs)
- [jitcode_lower.rs](/Users/al03219714/Projects/pypy/majit/majit-macros/src/jit_interp/jitcode_lower.rs)
- [jitcode.rs](/Users/al03219714/Projects/pypy/majit/majit-meta/src/jitcode.rs)

Rule:
- macro expansion must lower only to the runtime tracing seam below
- parity is not checked against token shape, only against resulting trace behavior

### 2. Runtime Tracing Seam

Input:
- symbolic interpreter events

Output:
- normalized IR trace

Current majit owners:
- [trace_ctx.rs](/Users/al03219714/Projects/pypy/majit/majit-meta/src/trace_ctx.rs)
- [recorder.rs](/Users/al03219714/Projects/pypy/majit/majit-trace/src/recorder.rs)
- [parity.rs](/Users/al03219714/Projects/pypy/majit/majit-meta/src/parity.rs)

Rule:
- this is the main parity gate for `pyjitpl`-style behavior
- tests should compare normalized trace lines, argument ordering, constants, and explicit guard fail-args

### 3. Execution Seam

Input:
- optimized trace

Output:
- compiled execution, guard exits, deadframes, GC-visible behavior

Current majit owners:
- `majit-opt`
- `majit-codegen`
- `majit-codegen-cranelift`
- `majit-gc`

Rule:
- parity here is behavioral: result values, guard exits, exceptions, resume, and GC interactions

## Test Layers

Every parity area should be tested in four layers.

1. Analyzer tests
- validate that translation-time source analysis finds opcode dispatch arms, helper classifications, trait impls, and generated helper code
- example: [majit-analyze/src/lib.rs](/Users/al03219714/Projects/pypy/majit/majit-analyze/src/lib.rs)

2. Macro lowering tests
- validate that supported interpreter syntax lowers into `JitCode` instead of emitting direct trace-recording code
- example: inline arithmetic lowered through [jitcode_lower.rs](/Users/al03219714/Projects/pypy/majit/majit-macros/src/jit_interp/jitcode_lower.rs)

3. Runtime seam tests
- validate that `TraceCtx` / `TraceRecorder` produce the expected normalized trace
- example helper: [parity.rs](/Users/al03219714/Projects/pypy/majit/majit-meta/src/parity.rs)

4. End-to-end parity tests
- validate interpreter result, optimized trace, compiled execution, and guard exits

## Mapping Rule

When comparing against RPython, use `RPython subsystem -> majit seam owner`
mapping, not `file -> file` mapping.

Examples:
- `pyjitpl` arithmetic execution
  - RPython: `execute(rop.INT_ADD, ...)`
  - majit: macro lowering in `jitcode_lower.rs` + generic `JitCode` execution in [jitcode.rs](/Users/al03219714/Projects/pypy/majit/majit-meta/src/jitcode.rs) + runtime seam in `TraceCtx`

- `pyjitpl` guard recording
  - RPython: `implement_guard_*`, `setfailargs()`
  - majit: `TraceCtx::record_guard*()` + normalized trace parity lines

- `history.TreeLoop`
  - RPython: loop/trace history objects
  - majit: `TraceRecorder` + `Trace`

## Claim Gating

`majit` should claim parity for a feature only when:

1. the owning seam is identified
2. unsupported semantics fail loudly
3. there is a normalized parity test at the seam
4. there is at least one end-to-end behavioral test for the same feature

### Current Claim Status

As of March 16, 2026, 1442+ tests pass across the workspace.

RPython parity tests exist for: resoperation, opencoder, history, threadlocal, tracelimit, jitprof, logger, fficall, memmgr, executor, rawmem, jitiface, bytearray.

| Feature | Seam | Gate 1 (owner) | Gate 2 (fail loudly) | Gate 3 (parity test) | Gate 4 (e2e test) |
|---|---|---|---|---|---|
| Arithmetic (add/sub/mul/div/mod) | Macro + Runtime | yes | yes | yes | yes |
| Bitwise ops / shifts | Macro + Runtime | yes | yes | yes | yes |
| Comparisons / booleanization | Macro + Runtime | yes | yes | yes | yes |
| Guard recording (GuardTrue/False) | Runtime | yes | yes | yes | yes |
| Helper calls (CallN/CallI/CallPureI) | Macro + Runtime | yes | yes | yes | yes |
| Auto-inferred calls (auto_calls) | Macro + Runtime | yes | yes | yes | yes |
| CallMayForce/ReleaseGil/Loopinvariant | Macro + Runtime | yes | yes | yes | yes |
| CallAssembler (typed dispatch) | Runtime + Backend | yes | yes | yes | yes |
| Translation-time analyzer / generated helpers | Build-time | yes | yes | yes | partial |
| Inline helpers (#[jit_inline]) | Macro | yes | yes | yes | yes |
| Resume/blackhole decode | Runtime | yes | yes | yes | yes |
| Virtual materialization | Runtime | yes | yes | yes | yes |
| Optimizer 8-pass pipeline | Execution | yes | yes | yes | yes |
| Float rewrites (7 simplifications) | Execution | yes | yes | yes | yes |
| IntDivision magic numbers | Execution | yes | yes | yes | yes |
| BridgeOpt guard elimination | Execution | yes | yes | yes | yes |
| HeapCache aliasing | Execution | yes | yes | yes | yes |
| WalkVirtual visitor | Execution | yes | yes | yes | yes |
| Virtual refs in optimizer | Execution | yes | yes | yes | yes |
| SIMD vector ops (I64X2/F64X2) | Execution | yes | yes | yes | yes |
| Loop versioning infrastructure | Execution | yes | yes | yes | yes |
| GC rewriter (nursery/write-barrier) | Execution | yes | yes | yes | yes |
| Safepoint/CompiledCodeRegistry | Execution | yes | yes | yes | yes |
| Frame-stack metadata for guards | Backend | yes | yes | yes | yes |
| Declarative driver lifecycle | Runtime | yes | partial | yes | partial |
| Multi-branch CFG lowering | Macro | yes | yes | yes | yes |
| Ref/float helper returns | Macro | yes | yes | yes | yes |
| Incremental GC semantics | Execution | yes | yes | yes | yes |
| JIT-GC hooks (pin/free/gc_step) | Execution | yes | yes | yes | yes |
| Multi-entry warmspot lifecycle | Runtime | yes | partial | yes | partial |
| FFI exchange buffer | Backend | yes | yes | yes | yes |
| Session-scoped deopt cache | Runtime | yes | yes | yes | yes |
| Multi-frame push/pop restore | Runtime | yes | yes | yes | yes |

## Current Arithmetic Coverage

As of March 16, 2026:

- `#[jit_interp]` lowers interpreter arms into `JitCode`, and a generic tracer in `majit-meta` executes that bytecode through an `MIFrame`-style machine with symbolic registers plus concrete int shadow state to produce the same trace shape for: add/sub/mul, div/mod (with IntDivision magic-number optimization in the optimizer), unary negation, bitwise ops, comparisons, shifts, integer literals, `if cond { 1 } else { 0 }` booleanization, internal `if/else` statement guards, register-valued `if/else` expressions, `state.pop()` / `state.push()` style interpreter code, config-aware residual void calls, helper-call lowering to `CallN` / `CallI` / `CallPureI`, bare `calls = { helper }` inference from helper attributes, opt-in `auto_calls = true` inference, hidden normalized call-target wrappers for mixed integer-like signatures, wrapper-backed explicit call-map policies, `#[jit_inline]` serialization, nested inline helpers with local policy inference, nested inline `CallPureI` / `CallN`, integer-helper arity `0..=8`, mixed `Ref` / `Float` helper arguments on the int/void helper-call subset, and the current raw inline-call subset with caller-callee register propagation
- the runtime tracing seam in `majit-meta` has normalized parity tests for arithmetic, constants, guards, explicit fail-args, div/mod, bitwise ops, shifts, and branch guards
- stable small-scope end-to-end parity harnesses exist in 14 test files under `majit-meta/tests/` covering inline arithmetic, shifts, comparisons, booleanized results, statement guards, branch expressions, branch-group behavior, config-aware residual calls, all helper-call variants (explicit/inferred/auto/wrapped/nested/inline), auto-inferred force/release-gil/loop-invariant helpers, mixed integer-like signatures, mixed Ref/Float helper arguments, typed raw JitCode CallMayForce*, token-backed raw JitCode CallAssembler*, driver marker rewriting, driver descriptors, and driver runtime parity
- `majit-analyze` now provides a build-time source-analysis seam for `pyre-mjit`: it parses multiple Rust source files, extracts opcode dispatch arms, resolves trait impls and helper classifications across files, and generates trace helper code plus JSON metadata during build. Current pyre analysis covers 40 opcode arms with 18 directly classified patterns and leaves the rest explicit residual/unclassified
- the optimizer pipeline now has comprehensive float rewrite coverage: FloatAdd(x, 0.0) and FloatAdd(0.0, x) identity, FloatSub(x, 0.0) identity, FloatMul(x, 1.0) and FloatMul(1.0, x) identity, FloatTrueDiv(x, 1.0) identity, FloatNeg(FloatNeg(x)) double-negation elimination, plus constant folding for FloatFloorDiv and FloatMod
- the optimizer also now has: CondCall optimization (dead-call elimination when condition is false, lowering to direct CallN when condition is true), PtrEq/PtrNe/InstancePtrEq/InstancePtrNe self-comparison and constant folding, CastPtrToInt/CastIntToPtr/CastOpaquePtr pass-through simplification, and GuardNoException removal after eliminated calls
- the recovery seam has complete resume/blackhole infrastructure: `EncodedResumeData` with RPython-style tagged numbering, all 5 virtual kinds, `ResumeLayoutSummary` with slot-level source detail, `CompiledExitLayout`/`CompiledExitArtifacts`/`CompiledTraceLayout` APIs, blackhole `BlackholeMemory` trait with all 31 Vec* opcodes handled, call/memory ops delegated, GUARD_NOT_FORCED no longer forcing virtuals, and `run_with_blackhole_fallback()` for compiled loop/bridge guard failure replay
- proc-macro-to-driver parity has advanced significantly: `#[jit_inline]` supports Int/Ref/Float return types with typed inline arg routing, `#[jit_module]` provides module-level auto-discovery of JIT-annotated helpers, `helpers = [...]` list syntax offers cleaner helper declaration, unsupported CFG patterns (standalone match, loop/while/for in arms) now emit `compile_error!`, JitDriver has `set_param()`/`get_stats()` runtime tuning, `JitHookInterface` hooks fire through real compile/bridge/error paths, and `majit-analyze` now provides a translation-time source-analysis/codegen path for `pyre-mjit`. Remaining gaps: multi-branch CFG lowering (detection exists but codegen does not lower complex match/loop arms into JitCode), analyzer classification breadth for general interpreter graphs, and RPython's full `warmspot.py` multi-entry lifecycle is not replicated
