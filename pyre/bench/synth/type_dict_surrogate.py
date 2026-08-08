# pyre-check: max-pypy-ratio=21
# pypy's exec time is pinned to the startup-subtraction floor here, so the
# ratio is not a measurement. The ceiling was 350 before the type-attribute
# fold: four times the slowest of an unfolded CI band of 21.5x-84.7x. Folded,
# twelve local readings across the three backends span 3.4x-7.0x, and the
# ceiling is three times the slowest of them. The unfolded band reached 2.13x
# the unfolded local reading, which puts a folded CI worst case near 15x, so
# the ceiling carries roughly 1.4x of headroom over it.
N = 200000

S = '\udcff'                  # lone low surrogate
H = 'a' + chr(0xd800) + 'b'   # embedded high surrogate


class Base:
    pass


class Sub(Base):
    pass


def main():
    setattr(Base, S, 1)
    setattr(Sub, H, 7)

    acc = 0
    i = 0
    while i < N:
        # hot read through the MRO via a lone-surrogate name
        acc = acc + getattr(Sub, S)
        # hot read of an embedded-surrogate name on the subclass itself
        acc = acc + getattr(Sub, H)
        i = i + 1

    # mutate after the loop, exercising mirror sync on a materialized __dict__
    d = Sub.__dict__
    setattr(Sub, S, 2)
    acc = acc + getattr(Sub, S)
    acc = acc + (1 if S in d else 0)
    delattr(Sub, S)
    acc = acc + (1 if S in d else 0)

    # iteration and length must account for surrogate keys
    keys = [k for k in Sub.__dict__ if k in (H, S)]
    acc = acc + len(keys)
    acc = acc + sum(1 for _ in Base.__dict__ if _ == S)

    print(acc)


main()
