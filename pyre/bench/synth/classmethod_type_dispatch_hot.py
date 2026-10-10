# pyre-check: max-pypy-ratio=3
# pyre-check: skip-cpython
# N is sized so pypy clears `FLOOR_GATE_MIN_BASELINE_S`.  At 300000
# iterations pypy's startup-subtracted user CPU sits under
# `EXEC_TIME_FLOOR_S`, the ratio prints with a `~`, and a ceiling is not
# applied.  120000000 iterations land pypy near 0.11s.  Local dynasm reads
# 1.3x; 3 leaves room for cranelift and a slower host.  cpython cannot run
# this many inside the reference timeout.
# 67f223fe51d (#2246) on main compiles one bridge: gf=201 br=1, t1/4:200
# then the bridge, t3/14:1. Nursery types split that TY_REF GUARD_VALUE
# (make_a_counter_per_value): dynasm 213/1 (t1/4:212), cranelift core
# 265/0 (t1/4:264, no hash reaches eagerness), wasm 201/1 matching main.
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
    while i < 120000000:
        total += Derived.scaled(i) + Base.scaled(i)
        i += 1
    print(total)


main()
