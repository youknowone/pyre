# pyre-check: gate=1
"""Module-dict cells stay out of the app-level gc inspector.

`MutableCell` is a `W_Root` with no typedef, so
`try_cast_gcref_to_w_root` returns None. `gc.get_referents` looks through
the cell and reports the value it holds. `gc.get_objects` and
`gc.get_rpy_roots` never return the cell as an object whose type is not a
type.
"""

import gc

X = 12345678
Y = object()
X = 7
X = 8

refs = gc.get_referents(globals())
bad = [r for r in refs if not isinstance(type(r), type)]
assert not bad, bad
assert any(r is Y for r in refs), refs

# `get_rpy_roots` is a PyPy `gc` extension; CPython has no raw root walk.
get_rpy_roots = getattr(gc, "get_rpy_roots", None)
if get_rpy_roots is not None:
    rpy_bad = [r for r in get_rpy_roots() if not isinstance(type(r), type)]
    assert not rpy_bad, rpy_bad

objs_bad = [o for o in gc.get_objects() if not isinstance(type(o), type)]
assert not objs_bad, objs_bad
