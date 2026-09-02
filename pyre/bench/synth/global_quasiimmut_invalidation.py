# pyre-check: max-pypy-ratio=5.4
# Ubuntu run 33279264115: 1.4-2.7x; the ceiling is twice the slowest,
# rounded up to one decimal place.
# pyre-check: skip-cpython
# cpython 0.21s vs pyre 0.07s (3.0x), and it is not gated on — only pypy is.
# Nested compiled loop + a conditional loop-carried store to a MODULE
# GLOBAL that is read after the (untaken) store.  The read-only global
# `x` folds to a constant in the primary loop under a
# GUARD_NOT_INVALIDATED keyed on the module dict's `version?`.  When the
# store rebinds `x`, the celldict bumps the version and flips the loop's
# invalidation flag, but the compiled GUARD_NOT_INVALIDATED must re-read
# that flag at runtime on every iteration so that a re-entry through any
# path (warm entry, CALL_ASSEMBLER, eval-breaker poll-deopt resume)
# observes the invalidation.  A guard that emits no runtime code (leaving
# only the warm-entry lookup to filter invalidated tokens) lets the periodic
# poll deopt re-enter the stale const-folded loop, and `x` reverts to its
# pre-store value for the rest of the run.  Function
# scope is unaffected (locals are never const-folded this way); the
# module-global read path is the one under test.
K = 5000
# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than this loop.
N = 23830300

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
