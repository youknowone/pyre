# pyre-check: max-pypy-ratio=4.4
# Ubuntu run 33279264115: 1.7-2.2x; the ceiling is twice the slowest,
# rounded up to one decimal place.
# pyre-check: skip-cpython
# Fitted between the two arms.  With every fold in place the three runners read
# 4.6x / 4.7x on darwin-arm64, 7.2x / 7.6x on ubuntu-24.04 and 9.2x / 10.0x on
# windows, at half the loop counts below.  The windows pair was marked `?`
# there: pypy's execution-only time sat under FLOOR_GATE_MIN_BASELINE_S, which
# is 0.15625s on a host whose CPU accounting advances in 1/64s ticks, and the
# pair cleared the ceiling of 8 it carried then only because `_compare_buffer`
# grants two ticks per unit of limit.  Doubling the work carries that baseline
# over the bar, so the number is judged rather than excused.  With
# PYRE_FBW_NO_SPECIALIZE=builtin_fold1,builtin_fold2 putting the same loops back
# on the residual, darwin-arm64 reads 52.9x and 61.4x -- eleven times the folded
# arm on that host.  The gate clears the highest folded reading by 20% and still
# sits an order of magnitude under the residual arm scaled to it.
#
# Those three-runner readings predate two changes to how these folds are
# recorded.  The `Int1` and `Float1` channels became elidable calls, so a fold
# whose operand does not change is hoisted out of the loop instead of called
# once per iteration.  And `min` / `max` over a pair of exact machine ints
# stopped emitting a call at all: what is guarded is the ordering of the two
# unboxed values, and under that ordering the answer is the winning operand's
# own reference.  darwin-arm64 reads 2.0x / 2.2x where it read 4.6x / 4.7x.
#
# pyre-check: spec-folds=builtin_fold1,builtin_fold2
# This fixture carried a wasm allowance of 13 while every folded builtin still
# left the trace module: a fold removes the frame force, the argument rooting,
# the execution-context resolution and the gateway binding, but each one still
# lowered to a call into a raw helper, and `abs(x)`'s helper is `(i64) -> f64`,
# a mixed signature the backend would not lower without a caller vouching for
# it.  31,998,958 crossings, 100% of them that one helper, 3.3s of a 5.0s run.
# Vouching it (`float_fold_helper_addrs`) takes the crossings to zero and the
# fixture to 1.2x, so the allowance is gone rather than lowered.
#
# The loop counts are twice what they once were, and stay that way: the
# denominator is small enough for startup subtraction to move it, and the
# subtraction error is a fixed number of milliseconds, so doubling the work
# halves its share.  Every recorded jit-stats counter is unchanged by it.
#
# `spec-folds` is the exact instrument neither ratio is.  Six loops sum into
# one number, so retiring one channel moves it by less than the span this
# fixture reads across the runners; the census gates each fold's coverage
# instead, and it reads the same on every host.
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
#                      arguments, so nothing is allocated at all.  A pair of
#                      exact machine ints does not reach the helper: the
#                      ordering guard picks the winner and the answer is that
#                      operand's own reference.
HASH_N = 32000000
ORD_N = 32000000
ABS_N = 32000000
MINMAX_N = 24000000


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
