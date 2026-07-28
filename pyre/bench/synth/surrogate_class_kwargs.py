# pyre-check: max-pypy-ratio=6
# Lone-surrogate keys survive the builtin dict / type kwargs and class
# namespace walks instead of crashing on the non-UTF-8 key:
#  - dict(**{surrogate: v}) and d.update(**{surrogate: v}) store the key
#  - type(name, bases, {surrogate: v}) puts a surrogate-named class attr
#  - class C(Base, **{surrogate: v}) forwards it to __init_subclass__

S1 = '\udc81'
S2 = '\udc84'


def main():
    # dict(**{surrogate}) keeps the key.
    d = dict(**{S1: 1, 'plain': 2})
    print('dict_s1', S1 in d, d[S1], len(d))

    # update(**{surrogate}) too.
    e = {}
    e.update(**{S2: 9})
    print('upd_s2', S2 in e, e[S2])

    # type(name, bases, ns) with a surrogate-named namespace entry.
    C = type('C', (), {S1: 5, 'plain': 6})
    print('type_attr', getattr(C, S1), getattr(C, 'plain'))

    # A surrogate class keyword reaches __init_subclass__.
    seen = []

    class Base:
        def __init_subclass__(cls, **kw):
            seen.extend(sorted(kw.keys(), key=lambda s: [ord(c) for c in s]))

    class Sub(Base, **{S1: 1, S2: 2}):
        pass

    print('subkw', [[ord(c) for c in s] for s in seen])


main()
