# set/frozenset methods with a fixed arity raise TypeError on a wrong
# positional-argument count instead of silently accepting it. The exact message
# text differs between CPython and PyPy, so only the exception type (which they
# agree on) is compared. Output verified against CPython/PyPy.
def kind(fn, args):
    try:
        fn(*args)
    except TypeError:
        return "TypeError"
    except KeyError:
        return "KeyError"
    else:
        return "ok"


def main():
    s = {1}
    f = frozenset({1})
    checks = [
        kind(s.symmetric_difference, ()),
        kind(s.symmetric_difference, ({2}, {3})),
        kind(f.issubset, ()),
        kind(f.issuperset, ({1}, {2})),
        kind(f.isdisjoint, ()),
        kind(f.copy, ({2},)),
        kind(s.add, ()),
        kind(s.discard, (1, 2)),
        kind(s.remove, ()),
        kind(s.pop, (1,)),
        kind(s.clear, (1,)),
        kind(s.symmetric_difference_update, ({2}, {3})),
    ]
    print(" ".join(checks))
    # correct-arity calls still work
    print(s.symmetric_difference({2}) == {1, 2}, f.isdisjoint({2}))
    s.add(2)
    s.discard(2)
    s.symmetric_difference_update({2})
    print(s == {1, 2})


main()
