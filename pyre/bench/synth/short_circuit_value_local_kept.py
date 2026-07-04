# Depth-1 kept-stack short-circuit with a BARE loop-varying local on the left
# and a loop-invariant constant on the right.
#
# `x = flag and CONST` lowers to a `goto_if_not` that keeps the falsy `flag`
# (the not-taken arm) live across the guard.  The hot path is the truthy arm
# (`flag` -> CONST), so a guard failure on the not-taken arm must restore
# `flag`, not the taken-path CONST.  Reading the guard-pc register file
# directly restores the stale CONST and silently miscompiles `kept_and`
# (over-counting toward `11 * N`: 2197063 instead of 1466663); the walk-level
# box mirror restores the kept `flag` correctly.
#
# `kept_stack_branch_depths.py` uses composite `(i % k)` left operands that
# const-fold differently and side-step this defect; each loop here pins the
# bare LOAD_FAST-left / const-right shape directly, in its own minimal
# function so the kept-stack guard resumes at depth 1 and the hot loop is the
# whole trace (a multi-shape loop dilutes the guard and hides the miscompile).
# Pure arithmetic -> deterministic checksum.
N = 200000


def kept_and():
    acc = 0
    i = 0
    while i < N:
        flag = i % 3
        x = flag and 11          # keeps falsy 0 on 1/3 of iters
        acc = acc + x
        i = i + 1
    return acc


def kept_or():
    acc = 0
    i = 0
    while i < N:
        flag = i % 3
        x = flag or 7            # keeps truthy flag on 2/3 of iters
        acc = acc + x
        i = i + 1
    return acc


def kept_not_and():
    acc = 0
    i = 0
    while i < N:
        flag = i % 3
        x = (not flag) and 11    # unary-not left, keeps False
        acc = acc + x
        i = i + 1
    return acc


def main():
    print(kept_and(), kept_or(), kept_not_and())


main()
