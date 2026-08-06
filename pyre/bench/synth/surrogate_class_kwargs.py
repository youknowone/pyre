# No `max-pypy-ratio`: this fixture compiles no loop -- its jitstats record
# `loops_compiled=0` -- so a pypy ratio compares two interpreters' startup
# rather than any generated code, and reads whatever the host's process
# spawn cost happens to be that run. The jitstats baselines gate it.
# Lone-surrogate keys survive the builtin dict / type kwargs and class
# namespace walks instead of crashing on the non-UTF-8 key:
#  - dict(**{surrogate: v}) and d.update(**{surrogate: v}) store the key
#  - type(name, bases, {surrogate: v}) puts a surrogate-named class attr
#  - class C(Base, **{surrogate: v}) forwards it to __init_subclass__

S1 = '\udc81'
S2 = '\udc84'


# The walks are repeated so the measured body is larger than the process
# startup floor; only the last round's answers are printed, so the output is
# identical to a single round.
REPEAT = 100


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
