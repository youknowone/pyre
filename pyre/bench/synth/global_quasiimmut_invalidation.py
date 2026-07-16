# pyre-check: max-pypy-ratio=150
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
N = 1000000

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
