# pyre-check: max-pypy-ratio=145
# The trip count now puts pypy above the startup-subtraction floor, so this
# ratio is a measurement rather than pyre divided by the floor constant. The
# ceiling is twice the slowest of the three backends observed unclamped
# (71.1x on wasm); the previous 45 was fitted against the clamp and fails.
# Inlined-callee shared-heap mutation parity, in both helper orderings.
#
# A tiny helper mutates a caller-owned list/instance inside a hot while-loop,
# so the tracer inlines the call and the mutating opcode (LIST_APPEND /
# STORE_ATTR) lands on a SHARED heap object. The final counts must match the
# iteration count exactly: a doubled side effect over-counts, a dropped one
# under-counts.
#
# `append_first` appends before bumping.  `store_first` reverses the order so
# the attribute store commits during recording BEFORE the deliberate
# list.append abort discards the trace: bump(c) traces through (its STORE_ATTR
# side effect commits during the inline concrete step), then push(acc, i) hits
# the append abort and the interpreter restarts the iteration from the
# unadvanced frame, re-running bump.  A recording attempt that aborts after
# the committed store must not leave a doubled side effect.
N = 2900000


def push(a, v):
    a.append(v)


class Counter:
    def __init__(self):
        self.n = 0


def bump(c):
    c.n = c.n + 1


def append_first():
    acc = []
    c = Counter()
    i = 0
    while i < N:
        push(acc, i)
        bump(c)
        i = i + 1
    print(len(acc))
    print(c.n)
    print(acc[0], acc[N // 2], acc[N - 1])
    print(sum(acc))


def store_first():
    acc = []
    c = Counter()
    i = 0
    while i < N:
        bump(c)
        push(acc, i)
        i = i + 1
    print(len(acc))
    print(c.n)
    print(acc[0], acc[N // 2], acc[N - 1])


append_first()
store_first()
