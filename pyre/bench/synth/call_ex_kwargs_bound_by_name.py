# `f(**mapping)` on a callee the inline takes: the keywords of an exact
# str-keyed dict are bound to the callee's parameters by name
# (fbw_bind_star_kwargs), with the dict's length pinned so no keyword is left
# over.  Each loop below is hot, so the binding is recorded on its first shape
# and every later shape has to leave through a guard: an unknown keyword, one
# too many or too few, a positional-only name, a keyword the star tuple
# already bound, a dict subclass, a non-str key, and a key set that changes
# after the loop compiled.  Each of those is a TypeError or a different
# binding the interpreter owns, and the printed messages and sums pin it.


def f2(a, b):
    return a + b


def f3(a, b=10, c=20):
    return a * 100 + b * 10 + c


def fp(a, /, b):
    return a - b


class D(dict):
    pass


def attempt(n, fn, mapping):
    r = None
    for i in range(n):
        try:
            r = fn(**mapping)
        except TypeError as ex:
            r = str(ex)
    return r


def main(n):
    d = {'a': 1, 'b': 2}
    e = {'b': 5}
    g = {'c': 7}
    s = 0
    for i in range(n):
        s += f2(**d) + f2(i, **e) + f3(**{'a': i}) + f3(1, **g) + fp(i, **e)
        s += f2(*(i,), **{'b': 3})
    print(s)
    for bad in ({'a': 1}, {'a': 1, 'b': 2, 'c': 3}, {'a': 1, 'x': 2},
                D(a=1, b=2), {1: 2}):
        print(attempt(n, f2, bad))
    r = None
    for i in range(n):
        try:
            r = f2(1, **d)
        except TypeError as ex:
            r = str(ex)
    print(r)
    print(attempt(n, fp, {'a': 1, 'b': 2}))
    r = None
    for i in range(n):
        dd = {'a': i, 'b': 1} if i < n // 2 else {'a': i, 'c': 1}
        try:
            r = f2(**dd)
        except TypeError as ex:
            r = str(ex)
    print(r)
    t = 0
    for i in range(n):
        dd = {'a': i, 'b': 1} if i < n // 2 else {'b': i, 'a': 1}
        t += f2(**dd)
    print(t)


main(3000)
