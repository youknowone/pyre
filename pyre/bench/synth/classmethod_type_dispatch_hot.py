"""Hot classmethod dispatch through a type receiver: `Type.cmethod(i)`.

Both `Derived.scaled` (inherited) and `Base.scaled` resolve to a classmethod
whose `cls` binds to the accessed class; the walker inlines the underlying
`__func__(cls, value)` in place of the descriptor-build + call residual pair.
Reading `cls.__name__` inside the body keeps a LOAD_ATTR in the inlined callee,
and the two distinct receiver classes exercise per-site type/version guards.
"""


class Base:
    @classmethod
    def scaled(cls, value):
        return value * 2 + len(cls.__name__)


class Derived(Base):
    pass


def main():
    total = 0
    i = 0
    while i < 50000:
        total += Derived.scaled(i) + Base.scaled(i)
        i += 1
    print(total)


main()
