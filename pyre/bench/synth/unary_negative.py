# pyre-check: max-pypy-ratio=4.8
# Ubuntu run 33279264115: 2-2.4x; the ceiling is twice the slowest,
# rounded up to one decimal place.
# pyre-check: spec-folds=unary_negative_descent,unary_negative_int
# The trip count puts pypy's execution above the startup-subtraction floor, so
# this ratio is a measurement. A ceiling far above the measurement would
# disable the gate at both ends, the derived floor included.
N = 38000000


# UNARY_NEGATIVE in a hot loop lowers to the `unary_negative(value)` HLOp →
# `residual_call_r_r(unary_negative_fn, ListR[value])` through
# opcode_ops::unary_negative_value (mirroring UNARY_INVERT / UNARY_NOT).
# Before the HLOp lowering the flow op `neg` reached the assembler with no
# builder mapping and any `-x` in a JIT-compiled loop panicked.
def main():
    acc = 0
    i = 0
    while i < N:
        x = -i
        acc = acc + x
        i = i + 1
    print(acc)


# UNARY_NEGATIVE on INT_MIN: -INT_MIN overflows the machine-int range, so
# descr_neg (intobject.py:628) takes the long branch and returns 2**63 as a
# W_LongObject.  The walker fold pins the operand with GUARD_VALUE and takes
# the _make_ovf2long tail, so the compiled loop must agree with the long result
# rather than wrapping back to INT_MIN.
#
# This is the operand the descent declines, which is why the header names both
# labels: `main` above is walked (`unary_negative_descent`) and this loop is
# folded (`unary_negative_int`).  Without the fold the promoted W_LongObject
# stays a loop argument, `compare_op_long` keeps its bigint call in the body,
# and the loop runs 2x slower.
def main_int_min():
    m = -9223372036854775807 - 1  # INT_MIN as a machine int
    acc = 0
    i = 0
    while i < N:
        if -m == 9223372036854775808:
            acc = acc + 1
        i = i + 1
    assert acc == N
    print(acc)


class Derived(int):
    def __neg__(self):
        return "NEG"

    def __pos__(self):
        return "POS"

    def __invert__(self):
        return "INV"


def check_guard_transitions():
    int_min = -9223372036854775807 - 1
    promoted = 0
    for value in [7] * 3000 + [int_min] * 3000:
        result = -value
        promoted += result == 9223372036854775808
    assert promoted == 3000

    # A subclass arrives after the exact-int trace is hot without a branch at
    # the operation site; each unary fold must leave through its class guard.
    for op, expected in (
        (lambda value: -value, "NEG"),
        (lambda value: +value, "POS"),
        (lambda value: ~value, "INV"),
    ):
        result = None
        for value in [7] * 3000 + [Derived(7)]:
            result = op(value)
        assert result == expected


main()
main_int_min()
check_guard_transitions()
