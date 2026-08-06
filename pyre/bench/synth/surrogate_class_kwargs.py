# pyre-check: max-pypy-ratio=20
# At the 100 it was recorded with, the loop never reached the JIT --
# `loops_compiled=0` on every backend -- and pypy's side sat on the
# execution floor, so the gate compared startup. At 3200 the loop compiles
# and pypy's own execution is a measurement. The ceiling is four times the
# slowest observed (5.0x), fitted on one host until the runners report.

S1 = '\udc81'
S2 = '\udc84'


# The walks are repeated so the measured body is larger than the process
# startup floor; only the last round's answers are printed, so the output is
# identical to a single round.
REPEAT = 3200


def main():
    for _ in range(REPEAT):
        # dict(**{surrogate}) keeps the key.
        d = dict(**{S1: 1, 'plain': 2})
        dict_s1 = (S1 in d, d[S1], len(d))

        # update(**{surrogate}) too.
        e = {}
        e.update(**{S2: 9})
        upd_s2 = (S2 in e, e[S2])

        # type(name, bases, ns) with a surrogate-named namespace entry.
        C = type('C', (), {S1: 5, 'plain': 6})
        type_attr = (getattr(C, S1), getattr(C, 'plain'))

        # A surrogate class keyword reaches __init_subclass__.
        seen = []

        class Base:
            def __init_subclass__(cls, **kw):
                seen.extend(sorted(kw.keys(), key=lambda s: [ord(c) for c in s]))

        class Sub(Base, **{S1: 1, S2: 2}):
            pass

        subkw = [[ord(c) for c in s] for s in seen]

    print('dict_s1', dict_s1[0], dict_s1[1], dict_s1[2])
    print('upd_s2', upd_s2[0], upd_s2[1])
    print('type_attr', type_attr[0], type_attr[1])
    print('subkw', subkw)


main()
