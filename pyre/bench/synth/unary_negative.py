# pyre-check: max-pypy-ratio=220
# The trip count puts pypy's execution above the startup-subtraction floor, so
# this ratio is a measurement. It reads higher than the clamped one it replaces
# rather than lower: a baseline pinned to the floor over-states pypy's work,
# so every ratio built on it was an under-estimate. The ceiling is twice the
# slowest of the three backends observed (109.5x on cranelift) -- the previous
# 110 sat on top of that reading and would fail on the run-to-run spread.
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
# W_LongObject.  generated_unary_int_value declines the int fast path at the
# concrete INT_MIN operand and traces the residual long-neg, so the compiled
# loop must agree with the long result rather than wrapping back to INT_MIN.
def main_int_min():
    m = -9223372036854775807 - 1  # INT_MIN as a machine int
    acc = 0
    i = 0
    while i < N:
        if -m == 9223372036854775808:
            acc = acc + 1
        i = i + 1
    print(acc)


main()
main_int_min()
