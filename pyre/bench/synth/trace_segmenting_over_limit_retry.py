# A loop body fat enough that one traced iteration runs past `trace_limit`, so
# the trace overflows before it can close. `blackhole_if_trace_too_long`
# (`pyjitpl.py:2812-2830`) finds no inlinable huge function here, so it calls
# `prepare_trace_segmenting` (`pyjitpl.py:2831-2859`), which marks the green key
# with the force-finish flag and asks for the next iteration to be traced again.
#
# On that retry the marked key is supposed to CUT: once the trace passes 0.8x
# `trace_limit` at a `jit_merge_point`, `pyjitpl.py:1617-1620` appends
# `GUARD_ALWAYS_FAILS`, compiles what it has, and blackholes with
# `ABORT_SEGMENTED_TRACE`. The cut arm is what turns the key's forever-retracing
# into compiled code.
#
# Without the cut, every retry runs into the same overflow and aborts again:
# `loops_aborted` read 64 with `bridges_compiled` 0 and `guard_failures` 11918.
# With it the key segments and the workload runs out of compiled arms instead.
#
# The 0.8x check only has something to fire at if a merge point is CROSSED while
# the trace sits in the 0.8x..1.0x band, so the body size and `trace_limit` are
# load-bearing together, not independently: at `trace_limit=200` the same body
# jumps the band in one crossing and never segments.  The band follows the RAW
# op count of one traced iteration, so a change in what the walker records for
# the body moves it: at 166 raw ops per iteration the cut fires for limits in
# 235..265 and nowhere above 270, and 250 sits in the middle of that.
#
# Which means the number tracks what one traced iteration COSTS in this tree,
# and has to be re-fit whenever that moves.  Swept against the recorded shape
# (`loops_compiled=2 bridges_compiled=4 loops_aborted=9 guard_failures=1008`),
# the window is 270..288 and 280 is its centre; 264 misses the band low and
# never segments (`loops_aborted=64 guard_failures=11922`), 294 crosses it high
# and settles on a worse three-loop shape.  It read 300 while the FOR_ITER
# receiver pin was spelled `ptr_eq` + `guard_true`; `guard_value` lets the
# optimizer constant-fold the class word out of the body, the iteration got
# shorter, and the whole window moved down with it.  Re-fit it by sweeping,
# not by nudging.
#
# CPython (the oracle) has no `pypyjit`; PyPy and pyre do. Guarding the import
# keeps the output identical across all three while the params only bind where a
# JIT exists. `set_param` rather than an environment variable because the wasm
# guest sees no environment.
try:
    import pypyjit

    pypyjit.set_param("trace_limit=250")
    pypyjit.set_param("threshold=20")
except ImportError:
    pass


def body(a, b, c, d, e):
    t = 0
    for _ in range(4):
        t += a * 3 + b - c
        t += (a + b) * (c - d) + e
        t ^= (t >> 3) + (a | b) + (c & d)
        t += a * b + c * d + e * 7
        t -= (a - b) * (c + d) - e
        t += (t & 0xFF) * 3 + a + b + c + d + e
    return t


s = 0
for i in range(4000):
    s += body(i, i + 1, i + 2, i + 3, i + 4)
print(s % 1000)
