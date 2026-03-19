# pyre vs PyPy Comparison Report

Date: 2026-03-19

## Summary

`pyre` has improved substantially on recursive `fib`, but it is still not at PyPy parity.

- Build status:
  - `cargo check -p pyre-main`: pass
  - `cargo build -p pyre-main --release`: pass
  - `cargo test -p pyre-jit test_eval_recursive_fib_compiles_function_entry_trace`: fail
- `fib_recursive` benchmark on the current release binary:
  - `pyre`: `user 0.23s`, but result is wrong
  - `pypy3`: `user 0.06s`

The current implementation is no longer failing because recursive traces never compile. However, it has now regressed functionally in both the debug test path and the release benchmark path.

## What Was Verified

Commands run locally:

```bash
cargo check -p pyre-main
cargo build -p pyre-main --release
cargo test -p pyre-jit test_eval_recursive_fib_compiles_function_entry_trace -- --nocapture --test-threads=1
/usr/bin/time -p ./target/release/pyre-main pyre/bench/fib_recursive.py
/usr/bin/time -p /opt/homebrew/bin/pypy3 pyre/bench/fib_recursive.py
MAJIT_LOG=1 ./target/release/pyre-main pyre/bench/fib_recursive.py
```

Observed results:

- `cargo check -p pyre-main`: pass
- `cargo build -p pyre-main --release`: pass
- `cargo test -p pyre-jit test_eval_recursive_fib_compiles_function_entry_trace`: fail
  - expected `fib(12) == 144`
  - actual value observed in the namespace was a large pointer-like integer
- `fib_recursive.py` in release mode is now also wrong
  - `pyre` printed `434365598085`
  - expected / `pypy3` printed `9227465`

## Current Recursive Trace Shape

The current hot recursive trace still looks like:

1. box `n - 1` / `n - 2` into Python int objects
2. call a force-able helper for the recursive call
3. guard `not_forced`
4. read integer payloads back from the boxed return values
5. perform `IntAddOvf`
6. box the final result again before `Finish`

Representative current trace pattern from `MAJIT_LOG=1`:

- `CallI(..., v34)` to box recursive arguments
- `CallMayForceI(..., frame, callable, boxed_arg)` for recursive calls
- `GetfieldRawI(v39)` / `GetfieldRawI(v61)` to read integer payloads back
- `IntAddOvf(v66, v69)`
- `CallI(..., v70)` to re-box the sum
- `Finish(v72)`

This is better than the older `create_frame -> CallAssembler -> drop_frame` shape, but it is still not the same as PyPy's recursive metatracing path. More importantly, the current boxed-result boundary is now functionally broken.

## How This Still Differs From PyPy

### 1. The active recursive path is still helper-based

The runtime recursive path in `pyre` still goes through helper calls:

- `pyre/pyre-jit/src/jit/helpers.rs`
- `pyre/pyre-jit/src/jit/state.rs`

In particular, known-function recursive calls are still emitted as `call_may_force_int` helper calls in the active path, not as a direct frame-switching interpreter trace-through path.

### 2. The more PyPy-like inline path exists, but is dead code

There is now an explicit attempt to move toward a PyPy-style inline interpreter stack:

- `pyre/pyre-jit/src/jit/state.rs`
  - `TraceFrameState.parent_fail_args`
  - `MIFrame`
  - `inline_trace_and_execute()`

However, the current build warns that these are unused:

- `parent_fail_args` field is never read
- `MIFrame` is never constructed
- `inline_trace_and_execute()` is never used

So the new PyPy-like direction is present in the source tree, but it is not yet the path that the JIT actually takes at runtime.

### 3. Raw-int finish was intentionally disabled again

In `majit-meta`, the raw-int finish protocol is currently disabled:

- `majit/majit-meta/src/pyjitpl.rs`

The current code explicitly forces boxed `Finish` values again instead of unboxing the final result. That matches the release trace, which now ends with boxed `Finish(v72)` instead of raw-int `Finish(v70)`.

This avoids raw-int leakage into frame locals, but it also reintroduces boxing cost on the recursive return path.

### 4. The create-frame boxing fold is present, but not wired in

There is a post-process pass meant to fold:

- `box(raw_int)`
- `create_frame(..., boxed_arg)`

into:

- `create_frame_raw_int(..., raw_int)`

That logic exists in:

- `majit/majit-meta/src/pyjitpl.rs`
  - `fold_box_into_create_frame()`

But it is currently dead code and not invoked in the finishing pipeline. The compiler warns about this directly.

That means one of the main remaining optimization ideas has not actually taken effect yet.

## Functional Regression

The biggest current problem is now correctness in both the debug test path and the release benchmark path.

This test now fails:

- `pyre/pyre-jit/src/eval.rs`
  - `test_eval_recursive_fib_compiles_function_entry_trace`

The failure is not just a policy mismatch. The `result` stored in the module namespace is no longer a valid boxed Python int in the tested configuration, and the release benchmark now shows the same class of corruption. That regression must be fixed before more performance work is trusted.

## Assessment

Current status is mixed:

- Good:
  - the tree builds again
  - recursive performance is much better than the old multi-second state
  - the codebase now contains a more PyPy-like recursive-inline direction
- Bad:
  - release `fib_recursive` no longer prints the correct result
  - the debug recursive regression test is also failing
  - the active runtime path is still helper-boundary recursion, not true trace-through recursion
  - recursive return boxing is back in the active release trace
  - `fold_box_into_create_frame()` is not active

## Recommended Next Steps

1. Fix the correctness regression in `test_eval_recursive_fib_compiles_function_entry_trace` and `fib_recursive.py` release output first.
2. Decide which recursive strategy is canonical:
   - keep helper-boundary recursion and optimize it aggressively
   - or make `inline_trace_and_execute()` the real runtime path
3. If helper-boundary recursion remains canonical, wire `fold_box_into_create_frame()` into the postprocess pipeline.
4. If true recursive inline is the goal, connect `inline_trace_and_execute()` to the active self-recursive hot path and remove the dead-code state once it is real.
5. Re-run the same `fib_recursive` benchmark after the correctness fix, because the current release speed cannot be trusted as a final baseline while the debug regression exists.

## Bottom Line

`pyre` is clearly closer to PyPy than it was before, but the current tree is in a transitional state:

- the old recursive path has been improved
- the more PyPy-like recursive path has been sketched
- but the new path is not actually live yet
- and a correctness regression has appeared while doing so

Right now the accurate description is:

`pyre` has regained buildability, but the current recursive result boundary is again functionally incorrect. It has not converged to PyPy's recursive metatracing model, and the current recursive test surface is not stable enough to call the work complete.
