# #171/#34: the orthodox list.append fold fires in function-entry (no-loop)
# helper traces, not only loop traces.  `push` is a no-loop helper called in a
# hot loop on two alternating receivers, so it traces from entry (header_pc==0)
# and its spare-capacity guard resume must reconstruct the alternating receiver
# correctly across the many realloc deopts — a wrong receiver box would route an
# append into the other list and corrupt the cross-checked sums below.
N = 200000


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
