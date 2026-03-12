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

As of March 12, 2026:

- `#[jit_interp]` no longer emits per-arm specialized tracing code for the supported subset; it now lowers interpreter arms into `JitCode`, and a generic tracer in `majit-meta` executes that bytecode through an `MIFrame`-style machine with symbolic registers plus concrete int shadow state to produce the same trace shape for add/sub/mul, div/mod, unary negation, bitwise ops, comparisons, shifts, integer literals, `if cond { 1 } else { 0 }` booleanization, internal `if/else` statement guards, register-valued `if/else` expressions, `state.pop()` / `state.push()` style interpreter code, config-aware residual void calls, helper-call lowering to `CallN` / `CallI` / `CallPureI`, bare `calls = { helper }` inference from helper attributes, opt-in top-level `auto_calls = true` inference for direct helper calls, `#[jit_inline]` helper serialization, nested inline helper calls with local sidecar-policy inference, nested inline `CallPureI`, nested inline `CallN`, integer-helper arity `0..=8` across the current helper-call subset, and the current raw inline-call subset with caller→callee int-register moves plus callee→caller return-register propagation
- the runtime tracing seam in `majit-meta` has normalized parity tests for arithmetic, constants, guards, explicit fail-args, div/mod, bitwise ops, shifts, and branch guards
- stable small-scope `jit_interp` end-to-end parity harnesses now exist in `majit-meta/tests/jit_interp_parity.rs`, `majit-meta/tests/jit_interp_branch_parity.rs`, `majit-meta/tests/jit_interp_config_parity.rs`, `majit-meta/tests/jit_interp_call_parity.rs`, `majit-meta/tests/jit_interp_auto_call_parity.rs`, `majit-meta/tests/jit_interp_inline_helper_parity.rs`, and `majit-meta/tests/jitcode_call_parity.rs` for inline arithmetic, shifts, comparisons, booleanized integer results, internal `if/else` statement guards, register-valued `if/else` expressions, branch-group `GuardTrue` / `GuardFalse` / `CloseLoop` behavior, config-aware residual calls / selector routing, helper-call lowering to `CallN` / `CallPureI` / `CallI` through explicit, bare inferred, and `auto_calls = true` paths, macro-generated inline helper lowering to inlined arithmetic ops, nested inline helpers plus nested inline `CallPureI` / `CallN` without mandatory local call maps, four-arg raw `JitCode` `CallPureI` / `CallN`, and four-arg proc-macro helper calls across explicit, auto-inferred, and inline-helper paths
- the recovery seam now has a narrow exception/deopt path: compiled deadframes can expose pending exception class/value, `GuardRecovery` carries that state plus the owning compiled `trace_id`, `ResumeData::encode()` now produces an RPython-style tagged numbering section with a shared const pool and compact sparse fail-arg numbering, those tagged resume sources reconstruct frame slots from fail-args, constants, virtual references, unavailable slots, and uninitialized content, pending field writes survive through the same encoded snapshot, blackhole execution can start with an already-pending exception, `blackhole_with_virtuals()` can surface materialized resume-data virtuals and pending writes, and `run_with_blackhole_fallback()` can replay the remainder of a supported guard failure from either the root compiled loop or an attached bridge through the blackhole seam
- broader proc-macro-to-driver parity is still incomplete: multi-branch control-flow graphs and fully automatic whole-module call discovery still do not exist, `#[jit_interp]` can infer helper policy from bare `calls = { helper }` entries backed by helper attributes, `#[jit_interp(auto_calls = true)]` can infer direct top-level helper calls without a call map, nested `#[jit_inline]` helpers can now infer direct helper calls without a local map, and integer-helper calls now work for arity `0..=8`, but `#[jit_inline]` still only covers simple int helper functions, richer helper signatures and helper calls beyond the current `0..=8` integer subset are still unsupported, richer exception paths are still partial, and non-arithmetic interpreter patterns still need dedicated fixture coverage
