# pyre-check: max-pypy-ratio=205
# pyre-check: skip-backends=cranelift
# cranelift runs the core build, which has no `pyre-module` and so no `math`.
N = 400000


def main():
    acc = 0
    i = 0
    while i < N:
        import math
        if math.pi > 3:
            acc = acc + 1
        i = i + 1
    print(acc)


main()
