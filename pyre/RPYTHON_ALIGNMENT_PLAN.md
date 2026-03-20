# Pyre RPython Alignment Plan

This plan is updated from the current tree as of 2026-03-20.
It is intentionally `pyre`-only. If a later step requires `majit`
changes again, stop there and escalate instead of half-wiring the path.

## Current Baseline

Verified on the current tree:

- `cargo test -p pyre-jit test_eval_recursive_fib_compiles_function_entry_trace -- --nocapture --test-threads=1`
  passes.
- `cargo build -p pyrex --release` passes.
- `./target/release/pyre pyre/bench/fib_recursive.py`
  returns `9227465`, roughly `user 0.12s`.
- `./target/release/pyre pyre/bench/inline_helper.py`
  returns the correct result, roughly `user 0.04s`.
- Local `pypy3` on the same machine is still faster:
  - `fib_recursive`: about `user 0.07s`
  - `inline_helper`: about `user 0.01s`

## What The Updated Code Actually Looks Like

### 1. Recursive correctness is currently stable

- self-recursive raw helper currently executes the callee frame through the
  pure interpreter force loop, not through direct compiled self-recursive
  re-entry:
  - `pyre/pyre-jit/src/call_jit.rs`
- function-entry tracing and raw-int finish are still active:
  - `pyre/pyre-jit/src/eval.rs`

This means the current recursive path is slower than the ideal RPython path,
but it is stable again.

### 2. Inline concrete execution still uses an ad-hoc result channel

- `pyre-interp/src/call.rs` uses `INLINE_HANDLED_RESULT: Cell<Option<PyObjectRef>>`
- `pyre-jit/src/jit/state.rs` still writes concrete inline results back through
  `set_inline_handled_result(...)`

This is not the final RPython-style design. It is an escape hatch that still
works, but it means inline result flow is not fully framestack-owned.

### 3. The biggest current performance gap is not recursive helper dispatch

The largest visible performance regression is currently `inline_helper`, not
`fib_recursive`.

From the optimized `inline_helper` trace:

- `New + SetfieldGc` boxed `W_Int` objects still survive in the final trace
- `LeavePortalFrame()` markers still survive in the final trace
- direct builtin/inline fast paths still return boxed objects through
  `trace_box_int(...)`

So the current hot inline path is still making object results escape instead of
staying as raw values long enough for the optimizer to remove the boxes.

### 4. Root self-recursive trace-through is still not the immediate next step

`MIFrame` / `inline_trace_and_execute` still exist in
`pyre-jit/src/jit/state.rs`, but the current code does not yet have a fully
framestack-owned concrete result and residual-call protocol.

Trying to force root self-recursive trace-through before fixing the escaping
inline result protocol is likely to produce another unstable half-state:

- helper-boundary recursion partially disabled
- framestack still incomplete
- concrete stack/result state diverging again

That is not aligned with RPython. RPython gets the representation and frame
ownership right first, then turns on deeper frame switching.

## Updated Priority Order

The previous plan over-emphasized root self-recursive trace-through too early.
The current code says the order should be:

1. stop hot inline results from escaping as boxed objects
2. make inline call/result ownership framestack-local
3. then re-open deeper frame-switch tracing
4. only after that, remove the remaining helper-boundary recursion

## Phase 0: Freeze The Current Baseline

Purpose:
- avoid mixing correctness regressions with performance work

Tasks:
- keep the current recursive test green
- keep both benchmark scripts runnable from `./target/release/pyre`
- save one representative `MAJIT_LOG=1` trace for:
  - `fib_recursive`
  - `inline_helper`

Completion criteria:
- recursive function-entry trace still compiles
- both benches still return correct answers

## Phase 1: Remove Escaping Boxed Int Results From Inline Hot Paths

This is the immediate highest-value step.

### Why this is first

Current optimized `inline_helper` trace still contains:

- `New()`
- `SetfieldGc(... ob_type ...)`
- `SetfieldGc(... intval ...)`
- later `GetfieldGcI(...)`

That means the `W_Int` result is escaping instead of being virtualized away.
This is the clearest current source of wasted work.

### Target

For pure int inline paths, keep values raw for as long as possible.
Only materialize a boxed `W_Int` at a boundary that truly requires an object.

### Files

- `pyre/pyre-jit/src/jit/state.rs`
- `pyre/pyre-jit/src/jit/helpers.rs`
- generated uses of `crate::jit::generated::trace_box_int(...)`

### Concrete work

1. Audit every `trace_box_int(...)` call site in `state.rs`
   - classify each call site as:
     - `must_box_now`
     - `can_stay_raw_in_inline_path`
2. Introduce an explicit raw-int inline result path for the direct helpers:
   - `direct_abs_value`
   - `direct_len_value` where the consumer can stay raw
   - `direct_minmax_value`
   - unary/binary/compare int fast paths
3. Delay boxing until one of these true boundaries:
   - store into a real object container that requires a `PyObjectRef`
   - residual helper call that still expects boxed objects
   - top-level return boundary
4. Re-measure the optimized `inline_helper` trace
   - `New + SetfieldGc` count must drop materially

### Completion criteria

- optimized `inline_helper` trace contains fewer boxed-int `New` nodes
- `inline_helper` user time drops materially from the current `~0.04s`

## Phase 2: Remove Portal Markers From Inline Hot Traces

### Why this is next

The optimized `inline_helper` trace still carries multiple
`LeavePortalFrame()` markers. Even if backend cost is low, they:

- increase trace size
- increase optimization work
- signal that inline bookkeeping is still too visible

### Target

Inline execution should not leave trace-level portal bookkeeping on the final
hot path unless it is semantically required.

### Files

- `pyre/pyre-jit/src/jit/state.rs`
- any inline call/return helper in `pyre-jit/src/call_jit.rs`

### Concrete work

1. identify where inline entry/leave markers are emitted today
2. distinguish:
   - real cross-portal transitions
   - inline-only bookkeeping markers
3. remove or suppress the inline-only markers
4. verify trace shape again

### Completion criteria

- optimized `inline_helper` trace no longer shows repeated inline-only
  `LeavePortalFrame()` markers

## Phase 3: Replace The Single-Slot Inline Result Channel

### Why this is now urgent

The current code reverted to:

- `INLINE_HANDLED_RESULT: Cell<Option<PyObjectRef>>`

That is simpler, but it is not RPython-like. It is also fragile for nested
inline/residual transitions.

### Target

Inline concrete result flow must become framestack-owned, not TLS-slot-owned.

### Files

- `pyre/pyre-interp/src/call.rs`
- `pyre/pyre-jit/src/jit/state.rs`

### Concrete work

1. remove dependence on the global single result slot for nested inline calls
2. move result ownership into `MIFrame`
3. make parent resume/result propagation explicit in the framestack
4. keep `call_callable_inline_residual(...)` but stop using TLS result passing
   for nested inline return values

### Completion criteria

- inline execution no longer depends on `set_inline_handled_result(...)`
  for nested call/return correctness

## Phase 4: Make Residual Calls Fully Framestack-Owned

### Why this still matters

There is already an explicit inline residual-call protocol:

- `pyre-interp/src/call.rs::call_callable_inline_residual`
- `pyre-jit/src/jit/state.rs::execute_inline_residual_call`

That was necessary, but it is still not the full ownership model.

### Target

Residual calls reached from inline execution should behave like explicit
framestack state transitions, not sidecar interpreter calls with borrowed state.

### Files

- `pyre/pyre-jit/src/jit/state.rs`
- `pyre/pyre-interp/src/call.rs`

### Concrete work

1. remove remaining hidden coupling between inline residual calls and outer
   interpreter call/result channels
2. keep concrete stack discipline local to the active `MIFrame`
3. audit nested residual call return handling

### Completion criteria

- nested inline residual calls no longer rely on interpreter-global state
- result propagation is explicit and local to the framestack

## Phase 5: Re-open Root Self-Recursive Trace-Through

Only after phases 1-4 are stable.

### Why later

At that point:

- inline results no longer escape as aggressively
- inline result ownership is framestack-local
- residual paths are no longer piggybacking on interpreter-global result state

Then root self-recursive frame-switch tracing becomes a structural extension,
not another half-wired bypass.

### Target

Allow self-recursive root calls back into `trace_through_callee()` /
`inline_trace_and_execute()` without helper-boundary recursion on the hot path.

### Files

- `pyre/pyre-jit/src/jit/state.rs`
- `pyre/pyre-jit/src/call_jit.rs`

### Concrete work

1. change the self-recursive `can_trace_through` gate
2. keep compiled/residual fallback only after inline limits
3. verify no stack divergence and no helper fallback on the hot recursive path

### Completion criteria

- `fib_recursive` still correct
- no `stack underflow`
- hot recursive trace no longer depends on the self-recursive force helper

## Phase 6: Remove Remaining Helper-Boundary Recursion

After root self-recursive trace-through is stable:

- remove `CallMayForceI(self_recursive_helper, ...)` from the hot recursive path
- reduce surviving callee frame materialization outside true residual boundaries

This is the point where the recursive path should start to look materially more
like PyPy's frame-switch behavior.

## Validation Matrix

Every phase should rerun at least:

- `cargo test -p pyre-jit test_eval_recursive_fib_compiles_function_entry_trace -- --nocapture --test-threads=1`
- `./target/release/pyre pyre/bench/fib_recursive.py`
- `./target/release/pyre pyre/bench/inline_helper.py`
- `MAJIT_LOG=1` trace capture for whichever hot path the phase targets

For performance-focused phases, compare against the current baseline:

- `fib_recursive`: ~`0.12 user`
- `inline_helper`: ~`0.04 user`

## Immediate Next Task

Start with Phase 1, not recursive frame switching:

- audit and reduce `trace_box_int(...)` escape points in `state.rs`
- make inline int results stay raw longer
- confirm that `New + SetfieldGc` count drops in the optimized
  `inline_helper` trace

That is the shortest path that is both:

- aligned with RPython/PyPy
- supported by the updated code we actually have today
