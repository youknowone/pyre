# pyre-check: max-pypy-ratio=96
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
