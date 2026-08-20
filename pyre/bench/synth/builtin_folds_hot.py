# pyre-check: max-pypy-ratio=8
# pyre-check: skip-cpython
# Fitted between the two arms as check.py itself reads them on darwin-arm64
# at load 18.  With every fold in place: 5.0x, 4.4x, 4.5x.  With
# PYRE_FBW_NO_SPECIALIZE=builtin_fold1,builtin_fold2 putting the same loops
# back on the residual: 52.9x and 61.4x.  The gate sits 60% above the first
# arm, which is room for a busy box, and still more than six times below the
# second, so what it reads is a lost fold.
#
# Every builtin the generic walker fold covers, one hot loop per channel.
#
# Without a fold, `hash(x)` / `ord(c)` / `abs(x)` / `min(a, b)` each reach the
# interpreter as `bh_call_fn(builtin, NULL, ...)`, and that residual costs the
# same for all of them: the frame force, the argument rooting, the execution
# context resolution and the gateway signature binding all run before the body
# does.  The fold emits a direct call into the builtin's raw helper, a guard on
# that channel's decline sentinel, and an inline `wrapint` / `wrapfloat` the
# optimizer can keep virtual.
#
# The ratio is the detector here: losing a fold changes no jit-stats counter,
# because the residual it falls back to compiles the same loop.
#
#   hash_int/hash_str  the `Int1` channel — an `i64` result and the
#                      `INT_FOLD_DECLINE` guard.
#   ord_str            the same channel on an operand whose acceptance is a
#                      length, not a type.
#   abs_int/abs_float  one builtin holding two rows, one per result channel.
#   min_max            the `Ref2` channel — the helper returns one of its own
#                      arguments, so nothing is allocated at all.
HASH_N = 16000000
ORD_N = 16000000
ABS_N = 16000000
MINMAX_N = 12000000


def run_hash_int():
    # `hash(int)` is the value itself, so this total is the same everywhere.
    total = 0
    x = 1234567
    for _ in range(HASH_N):
        total += hash(x)
    return total


def run_hash_str():
    # A string's hash is seeded per process, so the digest itself cannot be
    # printed.  Count the iterations that agree with the first one instead:
    # the fold still has to produce the digest, and the count is invariant.
    s = "specialize"
    first = hash(s)
    same = 0
    for _ in range(HASH_N):
        if hash(s) == first:
            same += 1
    return same


def run_ord():
    total = 0
    c = "q"
    for _ in range(ORD_N):
        total += ord(c)
    return total


def run_abs_int():
    total = 0
    x = -7
    for _ in range(ABS_N):
        total += abs(x)
    return total


def run_abs_float():
    total = 0.0
    x = -7.5
    for _ in range(ABS_N):
        total += abs(x)
    return total


def run_min_max():
    total = 0
    a = 3
    b = 9
    for _ in range(MINMAX_N):
        total += min(a, b) + max(a, b)
    return total


print(run_hash_int())
print(run_hash_str())
print(run_ord())
print(run_abs_int())
print(round(run_abs_float(), 6))
print(run_min_max())
