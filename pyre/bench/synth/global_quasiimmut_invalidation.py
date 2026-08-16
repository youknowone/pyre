# pyre-check: max-pypy-ratio=26
# pyre-check: skip-cpython
# The wasm allowance this carried (`max-wasm-ratio=6`, for the 5.2x and 5.1x
# reported on ubuntu-24.04) is gone because the ratio came down, not because
# the ceiling went up: the steady state no longer re-enters the invalidated
# loop, halving the executed wasm ops, and the ratio reads 2.9x here against
# 7.8x before.
# The pypy ceiling is twice the slowest ratio observed, 12.6x on the macos
# runner; the gate it replaces sat inside the run-to-run spread.
# cpython 0.21s vs pyre 0.07s (3.0x), and it is not gated on — only pypy is.
# Nested compiled loop + a conditional loop-carried store to a MODULE
# GLOBAL that is read after the (untaken) store.  The read-only global
# `x` folds to a constant in the primary loop under a
# GUARD_NOT_INVALIDATED keyed on the module dict's `version?`.  When the
# store rebinds `x`, the celldict bumps the version and flips the loop's
# invalidation flag, but the compiled GUARD_NOT_INVALIDATED must re-read
# that flag at runtime on every iteration so that a re-entry through any
# path (warm entry, CALL_ASSEMBLER, eval-breaker poll-deopt resume)
# observes the invalidation.  Previously the guard emitted no runtime
# code (only the warm-entry lookup filtered invalidated tokens), so after
# the periodic poll deopt the stale const-folded loop was re-entered and
# `x` reverted to its pre-store value for the rest of the run.  Function
# scope is unaffected (locals are never const-folded this way); the
# module-global read path is the one under test.
K = 5000
# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than this loop.
N = 10193192

x = 1
i = 0
s = 0

while i < N:
    j = 0
    while j < 2:
        j = j + 1
    if i == K:
        x = 100
    s = s + x
    i = i + 1

print(s)
