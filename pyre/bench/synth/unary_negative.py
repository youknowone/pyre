# pyre-check: max-pypy-ratio=8
# The trip count puts pypy's execution above the startup-subtraction floor, so
# this ratio is a measurement. The loop runs at parity: ubuntu 1.7x / 1.7x /
# 1.6x (dynasm / cranelift / wasm), macos 1.1x / 0.6x, windows 1.2x / 1.5x.
# The ceiling is twice the slowest of those, which puts the derived floor
# `min(1, ceiling / PERF_GATE_FLOOR_DIVISOR)` at 0.2x -- 2.5x under the fastest
# reading, so both bounds keep room. A ceiling far above the measurement would
# disable the gate at BOTH ends, the floor included.
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
