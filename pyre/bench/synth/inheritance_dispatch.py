# pyre-check: max-pypy-ratio=7
# pyre-check: max-wasm-ratio=5.9
# Method dispatch is a residual CALL on wasm. ubuntu-24.04 read 5.1x
# (1.12s / dynasm 0.22s). 5.9x is 5.1x plus WASM_RATIO_FIT_HEADROOM.
# Ubuntu run 33279264115: 1.5-3.5x; the ceiling is twice the slowest,
# rounded up to one decimal place.
# pyre-check: skip-cpython
# cpython 1.89s vs pyre 0.37s (5.1x on the ubuntu runner), and it is not
# gated on — only pypy is.
# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than this loop.
N = 24729800


class Base:
    def value(self, x):
        return x + 1


class Left(Base):
    def value(self, x):
        return x + 3


class Right(Base):
    def value(self, x):
        return x - 5


def main():
    objs = [Base(), Left(), Right()]
    i = 0
    acc = 0
    while i < N:
        acc = acc + objs[i % 3].value(i)
        i = i + 1
    print(acc)


main()
