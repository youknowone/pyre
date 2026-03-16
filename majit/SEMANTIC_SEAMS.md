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

Every parity area should be tested in three layers.

1. Macro lowering tests
- validate that supported interpreter syntax lowers into `JitCode` instead of emitting direct trace-recording code
- example: inline arithmetic lowered through [jitcode_lower.rs](/Users/al03219714/Projects/pypy/majit/majit-macros/src/jit_interp/jitcode_lower.rs)

2. Runtime seam tests
- validate that `TraceCtx` / `TraceRecorder` produce the expected normalized trace
- example helper: [parity.rs](/Users/al03219714/Projects/pypy/majit/majit-meta/src/parity.rs)

3. End-to-end parity tests
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

## Current Arithmetic Coverage

As of March 16, 2026:

- `#[jit_interp]` no longer emits per-arm specialized tracing code for the supported subset; it now lowers interpreter arms into `JitCode`, and a generic tracer in `majit-meta` executes that bytecode through an `MIFrame`-style machine with symbolic registers plus concrete int shadow state to produce the same trace shape for add/sub/mul, div/mod, unary negation, bitwise ops, comparisons, shifts, integer literals, `if cond { 1 } else { 0 }` booleanization, internal `if/else` statement guards, register-valued `if/else` expressions, `state.pop()` / `state.push()` style interpreter code, config-aware residual void calls, helper-call lowering to `CallN` / `CallI` / `CallPureI`, bare `calls = { helper }` inference from helper attributes, opt-in top-level `auto_calls = true` inference for direct helper calls, hidden normalized call-target wrappers for inferred/auto annotated helpers with mixed integer-like argument and return signatures, wrapper-backed explicit call-map policies for annotated helpers (`residual_void_wrapped`, `residual_int_wrapped`, `elidable_int_wrapped`), `#[jit_inline]` helper serialization, nested inline helper calls with local sidecar-policy inference, nested inline `CallPureI`, nested inline `CallN`, integer-helper arity `0..=8` across the current helper-call subset, and the current raw inline-call subset with caller→callee int-register moves plus callee→caller return-register propagation
- the runtime tracing seam in `majit-meta` has normalized parity tests for arithmetic, constants, guards, explicit fail-args, div/mod, bitwise ops, shifts, and branch guards
- stable small-scope `jit_interp` end-to-end parity harnesses now exist in `majit-meta/tests/jit_interp_parity.rs`, `majit-meta/tests/jit_interp_branch_parity.rs`, `majit-meta/tests/jit_interp_config_parity.rs`, `majit-meta/tests/jit_interp_call_parity.rs`, `majit-meta/tests/jit_interp_auto_call_parity.rs`, `majit-meta/tests/jit_interp_auto_call_family_parity.rs`, `majit-meta/tests/jit_interp_inline_helper_parity.rs`, and `majit-meta/tests/jitcode_call_parity.rs` for inline arithmetic, shifts, comparisons, booleanized integer results, internal `if/else` statement guards, register-valued `if/else` expressions, branch-group `GuardTrue` / `GuardFalse` / `CloseLoop` behavior, config-aware residual calls / selector routing, helper-call lowering to `CallN` / `CallPureI` / `CallI` through explicit, bare inferred, and `auto_calls = true` paths, auto-inferred `#[jit_may_force]` / `#[jit_release_gil]` / `#[jit_loop_invariant]` helper lowering for the current int/void subset, mixed integer-like inferred/auto helper signatures (`u32` / `usize` / `bool`) plus normalized non-`i64` integer-like returns through hidden call-target wrappers, mixed `Ref` / `Float` helper arguments on the supported int/void helper subset, wrapper-backed explicit call-map policies for annotated helpers, macro-generated inline helper lowering to inlined arithmetic ops, nested inline helpers plus nested inline `CallPureI` / `CallN` without mandatory local call maps, nested inline explicit wrapped helper calls, four-arg raw `JitCode` `CallPureI` / `CallN`, typed raw `JitCode` `CallMayForce*`, and token-backed raw `JitCode` `CallAssemblerI` / `CallAssemblerR` / `CallAssemblerF` / `CallAssemblerN`
- the recovery seam now has a narrow exception/deopt path: compiled deadframes can expose pending exception class/value, `CompileResult`, `GuardRecovery`, and `BlackholeRunResult` now also carry typed mixed-value exits decoded directly from backend fail-arg metadata plus backend `savedata`, `ResumeData::encode()` now produces an RPython-style tagged numbering section with a shared const pool and compact sparse fail-arg numbering, those tagged resume sources reconstruct frame slots from fail-args, constants, virtual references, unavailable slots, and uninitialized content, pending field writes survive through the same encoded snapshot, `ResumeLayoutSummary` now carries not just slot kinds but slot-level source detail (`compact fail-arg index`, `raw fail-arg position`, `constant`, `virtual index`) plus pending-write source detail, `CompiledExitLayout` now carries backend-rooted GC ref slots plus opaque force-token slots in addition to exit types and optional resume layout, `MetaInterp` now also exposes `CompiledExitArtifacts`, `CompiledTraceLayout`, backend-origin `source_op_index` / `trace_info`, aggregate guard and terminal exit layout queries, and `run_compiled_raw_detailed*()` so raw fast paths retain typed exits, layout, savedata, and exception state, terminal `FINISH`/`JUMP` ops now also have precomputed trace-local plus backend-origin exit layouts, blackhole execution can start with an already-pending exception, `blackhole_with_virtuals()` can surface materialized resume-data virtuals and pending writes, and `run_with_blackhole_fallback()` can replay the remainder of a supported guard failure from either the root compiled loop or an attached bridge through the blackhole seam
- broader proc-macro-to-driver parity is still incomplete: multi-branch control-flow graphs and fully automatic whole-module call discovery still do not exist, `#[jit_interp]` can infer helper policy from bare `calls = { helper }` entries backed by helper attributes, `#[jit_interp(auto_calls = true)]` can infer direct top-level helper calls without a call map, nested `#[jit_inline]` helpers can now infer direct helper calls without a local map, integer-helper calls now work for arity `0..=8` plus mixed integer-like inferred/auto argument and return signatures via normalized wrappers, mixed `Ref` / `Float` helper arguments are now supported on the current int/void traced helper-call subset, marker rewriting now accepts explicit nonstandard `driver` / `env` / `pc` / `stacksize` bindings while preserving the legacy default forms, those explicit marker forms can now also carry optional marker-local `; green1, green2, ...` tuples with consistency checks across merge/back-edge markers, `greens = [...]` back-edges now lower to structured `GreenKey` tuples on the runtime driver path, and the macro seam now rejects inconsistent marker bindings or a `can_enter_jit!` that appears before the first `jit_merge_point!` instead of silently rewriting a mismatched layout. `#[jit_driver]` also now carries optional `virtualizable = red_var` metadata, requires that virtualizable red to have `Ref` type when runtime types are supplied, can build checked runtime descriptors/structured green keys, implements a runtime-facing `DeclarativeJitDriver` trait, and a runtime `JitDriver` can attach that declarative descriptor to `TraceCtx` when tracing starts or build itself directly via `with_declarative(...)`. Declarative drivers now also validate red/live type mismatches at runtime before tracing begins, the generic `JitState` seam has a narrow typed live-value path so `Ref`/`Float` reds can survive trace-input registration plus direct compiled entry/exit on the supported subset, the raw compiled fast path now has both `MetaInterp::run_compiled_values()`, fully typed `run_compiled_with_values()`, and metadata-preserving `run_compiled_raw_detailed*()` variants, and `JitDriver::back_edge()` now consumes `extract_live_values()` plus `restore_values()` on compiled fast paths instead of forcing mixed values through the legacy int-only restore seam. The same seam now exposes three virtualizable layers: `sync_named_virtualizable_before_jit` / `sync_named_virtualizable_after_jit` always fire when declarative virtualizable metadata is present, `sync_virtualizable_before_jit` / `sync_virtualizable_after_jit` still receive the richer `VirtualizableInfo` path when that metadata is configured, and interpreters can now opt into runtime-managed heap synchronization by exposing a virtualizable heap pointer plus box import/export hooks while the runtime performs the common `VirtualizableInfo`-driven heap read/write and token-clear work. The runtime driver also now has a narrow deopt-aware path, `run_compiled_with_blackhole_fallback_*`, which applies `restore_guard_failure_values()` plus the same post-JIT virtualizable sync seam on `Jump`, `GuardFailure`, and `Abort` exits from supported blackhole/recovery cases, decodes raw exit words back into typed `Ref`/`Float` reds for the common declarative subset before calling interpreter restore hooks, can automatically project reconstructed resume frames through typed per-frame and metadata-aware `JitState` hooks, can replay pending field and array writes through the common `PendingFieldWriteLayout` seam, can still fall back to custom `restore_reconstructed_frames()` overrides for interpreter-specific frame stacks, preserves nested virtual references inside `MaterializedVirtual` graphs instead of erasing them to placeholder raw values, lets interpreters materialize virtual `Ref` reds through `JitState::materialize_virtual_ref()` / `materialize_virtual_ref_with_refs()`, and now also exposes `CompiledExitLayout`, `CompiledExitArtifacts`, and richer `ResumeLayoutSummary` metadata so caller-stack reconstruction code can inspect exit slot types, rooted GC ref slots, opaque force-token slots, frame slot-source kinds, virtual kinds, pending-write kinds, and backend `source_op_index` / `trace_info` without eagerly reconstructing the whole state. Annotated helpers can now also use wrapper-backed explicit call-map policies when the caller wants an explicit `CallN` / `CallI` / `CallPureI` kind, including from nested `#[jit_inline(calls = { ... })]` helpers, but `#[jit_inline]` still only covers simple int helper functions, raw explicit call-map entries without the wrapped policy variants still assume the old direct `i64` ABI, ref/float helper return signatures are still unsupported, richer exception paths are still partial, full declarative driver/runtime lifecycle parity is still missing, and non-arithmetic interpreter patterns still need dedicated fixture coverage
