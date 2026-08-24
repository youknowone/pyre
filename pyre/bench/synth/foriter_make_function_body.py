# MAKE_FUNCTION plus SET_FUNCTION_ATTRIBUTE in a hot FOR_ITER body. The default
# value forces the companion attribute initializer onto the definition path.
#
# Sized against FLOOR_GATE_MIN_BASELINE_S, which is what decides whether the
# wasm/dynasm ratio is judged at all: below it `run_synthetic_bench` declines
# the gate and prints the reading with a `~`, above it the ceiling fires. At
# the previous 20000 this fixture's dynasm execution-only time landed on both
# sides of that line from run to run, so which way the gate went was a property
# of the host rather than of the backend -- and the readings that fell on the
# judged side (5.1x, 5.5x, 5.8x over six reps) were far BETTER than the ones
# excused beside them (~15x), because a denominator that small is mostly its
# own subtraction error. A pre-change binary failed at the same rate, so
# nothing about the JIT was being measured. Here dynasm's total lands at
# 0.38-0.47s across the three legs, so its execution-only time clears the
# darwin/linux line six times over, and the windows one -- ten times larger,
# because that host's CPU accounting advances in 1/64s ticks -- by two.
#
# Judged rather than declined, the wasm ratio reads 1.1-1.6x over eleven runs
# against the 4x default. The 5-6x it used to read whenever it landed on the
# judged side was the subtraction error, not the backend.
#
# The pypy column stays informational: this fixture states no
# `max-pypy-ratio`, so `synth_perf_gate` returns None and no ratio against
# pypy is applied. pypy runs the whole loop in hundredths of a second at any
# size worth running, so that column could not be gated here anyway.
N = 8000000


def main():
    total = 0
    for i in range(N):

        def add(value=i):
            return value + 1

        total += add()
    print(total)


main()
# Expected: 32000004000000
