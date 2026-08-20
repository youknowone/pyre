# pyre-check: max-pypy-ratio=8
# The trip count puts pypy's execution above the startup-subtraction floor, so
# this ratio is a measurement. It reads higher than the clamped one it replaces
# rather than lower: a baseline pinned to the floor over-states pypy's work,
# so every ratio built on it was an under-estimate. The ceiling is twice the
# slowest of the backends observed.
#
# 220 came from a 109.5x cranelift reading; the loop now runs at parity, so the
# derived floor is the bound that noticed. A ceiling that far above its
# measurement is not a loose bound but a disabled one at BOTH ends: the floor is
# `min(1, ceiling / PERF_GATE_FLOOR_DIVISOR)`, so 220 pinned it at parity, and
# cranelift reading 0.5x on macos failed it from below.
#
# Observations behind 8 -- ubuntu 1.7x / 1.7x / 1.6x (dynasm / cranelift /
# wasm), macos 1.1x / 0.6x, windows 1.2x / 1.5x, and 2.5x / 3.9x on a loaded
# dev box where pypy's own execution lands in the thin band and inflates every
# ratio built on it. Twice the slowest of those puts the floor at 0.2x, which
# is 2.5x under the fastest reading -- both bounds keep room, which is the
# whole point of deriving one from the other.
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
