# CPython-suite gap: SystemExit tests check `code` once, never from a loop hot
# enough for a JIT constructor specialisation to take over.
# parity-tests reason: pyre specialises the SystemExit constructor in traced
# code, so `code` is written by trace IR rather than by the runtime __init__.

# `interp_exceptions.py:993-998 W_SystemExit.descr_init`: no argument leaves
# `code` at the class default, one argument becomes `code` verbatim, and
# several become the args tuple. A constructor specialisation that emits the
# `code` store itself has to reproduce all three, for the builtin and for a
# subclass that adds no `__init__`, and for the two- and three-element tuple
# shapes separately because a two-element tuple has its own storage layout.

N = 3000


class MyExit(SystemExit):
    pass


def shapes(factory, n):
    """Every `code` the constructor produces, over a loop the JIT compiles."""
    seen_none = 0
    seen_int = seen_str = 0
    pairs = set()
    triples = set()
    for i in range(n):
        assert factory().code is None
        seen_none += 1

        assert factory(i).code == i
        seen_int += 1

        assert factory("bye").code == "bye"
        seen_str += 1

        pairs.add(factory(i, "two").code)
        triples.add(factory(i, "three", 3.5).code)
    return seen_none, seen_int, seen_str, pairs, triples


for factory in (SystemExit, MyExit):
    name = factory.__name__
    none_hits, int_hits, str_hits, pairs, triples = shapes(factory, N)
    assert (none_hits, int_hits, str_hits) == (N, N, N), name
    assert pairs == {(i, "two") for i in range(N)}, name
    assert triples == {(i, "three", 3.5) for i in range(N)}, name
    assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs), name
    assert all(isinstance(t, tuple) and len(t) == 3 for t in triples), name

# `args` is stamped by the base constructor and must stay independent of the
# `code` rule: a lone argument is still a one-element args tuple.
for factory in (SystemExit, MyExit):
    assert factory().args == ()
    assert factory(7).args == (7,)
    assert factory(7, "x").args == (7, "x")

# Assigning `code` afterwards must win over whatever the constructor stored.
for factory in (SystemExit, MyExit):
    e = factory(1, 2)
    e.code = "replaced"
    assert e.code == "replaced", factory.__name__
    assert e.args == (1, 2), factory.__name__

print("OK")
