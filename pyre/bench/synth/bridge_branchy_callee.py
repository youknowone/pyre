# pyre-check: max-pypy-ratio=10
# A branch inside an inlined callee (gh#343). `compute_branch` guard-fails on the
# rare arm (`x % 7 == 0`); without a compiled bridge for the inlined callee's
# continuation every crossing deopts to the blackhole (~34x slower on the native
# backends). The carrier-resume sub-walk inlines the branchy continuation's
# nested int-arith chain (add/mul/square) instead of residualizing it, so the
# bridge for the inlined callee's branch arm compiles.
#
# It mirrors inline_helper.py's add/mul/square/compute call chain, but compute()
# here branches on data — the corpus otherwise never exercises a branch inside an
# inlined callee, the one shape that still failed. The gate is against pypy (which
# inlines the branchy callee and bridges the cold arm), so a decline that keeps
# every crossing interpreted balloons the ratio well past the limit.

N = 8_000_000
M = 1_000_000_007


def add(a, b):
    return a + b


def mul(a, b):
    return a * b


def square(x):
    return mul(x, x)


def compute_branch(x):
    if x % 7 == 0:
        return add(x, x)
    return add(square(x), x)


def drive_branch(n):
    s = 0
    i = 0
    while i < n:
        s = add(s, compute_branch(i)) % M
        i = add(i, 1)
    return s


print(drive_branch(N))
