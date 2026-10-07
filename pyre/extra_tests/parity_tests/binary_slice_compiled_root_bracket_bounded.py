# CPython-suite gap: compiled BINARY_SLICE root-bracket depth is a pyre JIT
# concern; CPython has no TLS shadow stack.
# parity-tests reason: descending `binary_slice_values_inner` must not grow
# the shadow stack by three slots per iteration (publish_roots without close).
# Descent signal: pyre/bench/synth/binary_slice_descended_list.py (exact-int
# `xs[a:b]` while-loop) plus its `.jitstats`; this file checks RSS on the
# same BINARY_SLICE shape after that loop has compiled.

import gc
import resource
import sys

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass


def rss_bytes():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage
    return usage * 1024


def slice_loop(n, xs, start, stop):
    total = 0
    for _ in range(n):
        total += len(xs[start:stop])
    return total


xs = list(range(32))
start, stop = 1, 8
assert slice_loop(8000, xs, start, stop) == 7 * 8000
gc.collect()
before = rss_bytes()
assert slice_loop(400000, xs, start, stop) == 7 * 400000
after = rss_bytes()
# BINARY_SLICE `xs[start:stop]` (not a folded slice constant). 400k * 3 * 8
# = 9.6 MiB if inner `publish_roots` is never popped. Peak RSS stays well
# below that once the compiled residual's RootScope closes.
assert after - before < 8 * 1024 * 1024, (before, after, after - before)

print("OK")
