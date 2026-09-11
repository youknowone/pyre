# pyre-check: max-pypy-ratio=7
# pyre-check: max-wasm-ratio=6.3
# leftover-empty vable tails now GETFIELD the live/baked field list
# (`compile.py patch_new_loop_to_load_virtualizable_fields`). dynasm
# turns those into native loads; wasm emits them as guest ops. darwin-arm64
# measured 5.4x against dynasm; 6.3x is that reading plus
# WASM_RATIO_FIT_HEADROOM (15%).
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
