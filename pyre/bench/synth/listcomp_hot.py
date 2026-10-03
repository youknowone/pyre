# pyre-check: max-pypy-ratio=134.5
# dynasm 40.8-53.5x, cranelift 44.8-67.2x; the ceiling is twice the slower,
# rounded up to one decimal place.
N = 200000


def build(n):
    # An inlined list comprehension compiles to LOAD_FAST_AND_CLEAR
    # (isolating the `j` iteration variable) around a hot FOR_ITER +
    # LIST_APPEND body.  Before LOAD_FAST_AND_CLEAR was lowered, its
    # abort_permanent marker declined the whole comprehension loop.
    return [j & 3 for j in range(n)]


def main():
    total = 0
    for _ in range(20):
        total += sum(build(N))
    print(total)


main()
