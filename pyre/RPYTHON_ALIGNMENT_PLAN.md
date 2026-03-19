# Pyre RPython Alignment Plan

This document captures the pyre-only work needed to move recursive tracing
from helper-boundary recursion toward PyPy/RPython-style frame-switch tracing.

## Current Stable Baseline

- `cargo test -p pyre-jit test_eval_recursive_fib_compiles_function_entry_trace -- --nocapture --test-threads=1`
  passes.
- `cargo build -p pyre-main --release` passes.
- `./target/release/pyre-main pyre/bench/fib_recursive.py`
  returns `9227465`, with roughly `user 0.24s` on the current machine.
- `./target/release/pyre-main pyre/bench/inline_helper.py`
  returns the correct result, with roughly `user 0.02s`.

## Current Hard Blocker

Root self-recursive trace-through is still disabled on purpose.

When root self-recursive `can_trace_through` is re-enabled, concrete execution
eventually falls through a residual/known-function call path that is still
owned by the generic interpreter call protocol instead of the inline
framestack. The concrete state then diverges from the inline framestack state,
and the system fails with:

- `TypeError: stack underflow during interpreter opcode`
- or `SIGSEGV`

The failing boundary consistently shows up through:

- `pyre-jit/src/jit/state.rs`: self-recursive inline path
- `pyre-runtime/src/runtime_ops.rs`: `jit_call_known_function_*`
- `pyre-jit/src/call_jit.rs`: `jit_call_user_function_from_frame`

In short: tracing can switch frames, but concrete residual execution still
cannot.

## Design Target

Match PyPy/RPython more closely:

- tracing and concrete execution both use a frame stack
- call/return/residual-call transitions are owned by that framestack
- self-recursive hot paths do not escape back to helper-boundary recursion
  once tracing has entered inline execution

The target seam is `MIFrame` / `inline_trace_and_execute` in
`pyre-jit/src/jit/state.rs`.

## Ordered Work Plan

### Phase 1. Make Concrete Residual Calls Framestack-Owned

Goal:
- residual calls reached during inline concrete execution must no longer route
  back through generic `call_user_function()` behavior

Required work:
- introduce an explicit "inline framestack residual call" protocol in
  `pyre-interp/src/call.rs`
- keep it separate from:
  - normal interpreter calls
  - generic JIT helper-boundary calls
- route only framestack-owned residual calls through it

Success criteria:
- nested inline execution can perform a residual user-function call without
  touching the outer interpreter stack protocol

### Phase 2. Extend MIFrame to Own Concrete Call/Return State

Goal:
- `MIFrame` must own enough concrete state to enter, resume, and return across
  nested calls without falling back to TLS/result-slot tricks

Required work:
- make the framestack carry explicit resume/result state
- stop relying on implicit concrete-result handoff for nested residual paths
- make parent resume deterministic and local to the framestack

Success criteria:
- inline concrete execution no longer depends on generic interpreter call
  re-entry to move results between parent and child frames

### Phase 3. Re-enable Root Self-Recursive Trace-Through

Goal:
- allow self-recursive root calls back into `trace_through_callee()`

Required work:
- change the self-recursive `can_trace_through` gate in
  `pyre-jit/src/jit/state.rs`
- keep the fallback policy the same once recursion exceeds inline limits:
  residual/compiled boundary only after the framestack can own it safely

Success criteria:
- `fib_recursive` passes with root self-recursive trace-through enabled
- no `stack underflow`, no `SIGSEGV`

### Phase 4. Remove Helper-Boundary Recursion from the Hot Recursive Path

Goal:
- the hot recursive `fib` trace should stop going through
  `CallMayForceI(jit_force_self_recursive_call_raw_1, ...)`

Required work:
- verify that recursive hot traces stay in frame-switch tracing
- only leave through explicit residual/compiled boundaries after inline limits

Success criteria:
- the optimized trace no longer shows helper-boundary recursion for the hot
  recursive path

### Phase 5. Resume Performance Work

Only after phases 1-4 are stable:

- push virtual `W_Int` further so inline returns do not escape as boxed ints
- reduce callee frame materialization that still survives outside the
  frame-switch path
- trim remaining call/namespace bookkeeping on the recursive hot path

## Guardrails

- Do not change `majit/*` as part of this plan.
- Keep `fib_recursive` correctness green at every step.
- Do not leave half-enabled root self-recursive trace-through in the tree.
- If a change requires helper-boundary recursion to stay active, keep the
  existing self-recursive root gate disabled.

## Immediate Next Task

Implement Phase 1:

- define an explicit framestack residual-call protocol in `pyre-interp`
- wire `inline_trace_and_execute` to use it for concrete nested residual calls
- keep root self-recursive trace-through disabled until that path is proven
  correct
