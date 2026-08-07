# pyre-check: max-pypy-ratio=125
# The trip count now puts pypy above the startup-subtraction floor, so this
# ratio is a measurement rather than pyre divided by the floor constant, and
# it reads higher than the clamped one it replaces -- a baseline pinned to the
# floor over-states pypy's work and so under-states the ratio. The ceiling is
# twice the slowest backend observed unclamped (61.9x on cranelift); the
# previous 61 was fitted against the clamp and sat under that reading.
# #171/#34: the orthodox list.append fold fires in function-entry (no-loop)
# helper traces, not only loop traces.  `push` is a no-loop helper called in a
# hot loop on two alternating receivers, so it traces from entry (header_pc==0)
# and its spare-capacity guard resume must reconstruct the alternating receiver
# correctly across the many realloc deopts — a wrong receiver box would route an
# append into the other list and corrupt the cross-checked sums below.
N = 1500000


def push(a, v):
    a.append(v)


def main():
    xs = []
    ys = []
    i = 0
    while i < N:
        push(xs, i)
        push(ys, -i)
        i = i + 1
    ok = (
        len(xs) == N
        and len(ys) == N
        and xs[0] == 0
        and ys[1] == -1
        and xs[N - 1] == N - 1
        and ys[N // 2] == -(N // 2)
    )
    print(sum(xs) + sum(ys), len(xs) + len(ys), ok)


main()
