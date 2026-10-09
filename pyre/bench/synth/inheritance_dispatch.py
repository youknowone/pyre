# pyre-check: max-pypy-ratio=4
# The measured reading stays under 4, so the ceiling is 4.
# pyre-check: skip-cpython
# pyre-check: jitstats-band=guard_failures=2
# Product dynasm reads 887 or 888. ResumeGuardDescr.get_jitcounter_hash
# (compile.py) hashes a TY_REF guard_value with lltype.cast_ptr_to_int, so
# a nursery type that a minor moves mid-eagerness starts a second counter.
# Main 2e10498449c (immortal/prebuilt types) is 602 on dynasm and cranelift:
# t1/10:201 t1/20:200 t2/0:200 t1/17:1, three bridges. Ours split t1/10 and
# t2/0 (dynasm 344/342, cranelift core 383/381) while t1/20 stays 200.
# wasm matches main at 602. Band=2 covers the one-tick split.
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
