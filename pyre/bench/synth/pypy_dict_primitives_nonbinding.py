"""__pypy__.reversed_dict / move_to_end / objects_in_repr are non-binding
builtins (interp-level), mirroring PyPy's interp_dict / interp_magic placement.

All checks use ``assert`` so every backend prints only the final ``OK``: the
collections.OrderedDict section runs everywhere; the ``__pypy__``-direct section
(the module exists only on pyre/PyPy) is guarded and adds no distinguishing
output.
"""

from collections import OrderedDict

# --- OrderedDict behavior (CPython 3.14, PyPy, pyre) ---
od = OrderedDict([("a", 1), ("b", 2), ("c", 3)])

# reversed() -> reversed_dict
assert list(reversed(od)) == ["c", "b", "a"]

# move_to_end last=True, then last=False
od.move_to_end("a")
assert list(od) == ["b", "c", "a"]
od.move_to_end("c", last=False)
assert list(od) == ["c", "b", "a"]

# a missing key raises KeyError
try:
    od.move_to_end("zzz")
    raise AssertionError("move_to_end on a missing key must raise")
except KeyError:
    pass

# recursive-repr guard -> objects_in_repr (repr string differs across interps,
# only the "..." sentinel is asserted)
d = OrderedDict()
d["self"] = d
assert "..." in repr(d)
d.clear()

# --- __pypy__ direct + non-binding (pyre / PyPy only; asserts only) ---
try:
    import __pypy__
except ImportError:
    __pypy__ = None

if __pypy__ is not None:
    rd = __pypy__.reversed_dict
    mte = __pypy__.move_to_end
    oir = __pypy__.objects_in_repr

    # (1) non-binding: stored as a class attribute it is NOT rebound to a bound
    # method (a binding function would yield a fresh bound method != the func),
    # and calling through the instance does not prepend the receiver as self.
    # (The type name is intentionally not asserted: CPython reports
    # 'builtin_function_or_method', PyPy 'builtin_function'.)
    class Holder:
        a_rd = rd
        a_mte = mte
        a_oir = oir

    h = Holder()
    assert h.a_rd is rd
    assert h.a_mte is mte
    assert h.a_oir is oir
    assert list(h.a_rd({"a": 1, "b": 2})) == ["b", "a"]

    # (2) direct behavior on a plain dict (subclass-bypassing backing path)
    plain = {"x": 1, "y": 2, "z": 3}
    assert list(rd(plain)) == ["z", "y", "x"]
    mte(plain, "x")
    assert list(plain) == ["y", "z", "x"]
    mte(plain, "z", False)
    assert list(plain) == ["z", "y", "x"]
    assert plain == {"z": 3, "y": 2, "x": 1}
    try:
        mte(plain, "nope")
        raise AssertionError("move_to_end miss must raise KeyError")
    except KeyError:
        pass
    try:
        rd([1, 2, 3])
        raise AssertionError("reversed_dict of a non-dict must raise TypeError")
    except TypeError:
        pass

    # (3) objects_in_repr is a stable identity_dict (id-keyed, unhashable ok)
    r1 = oir()
    r2 = oir()
    assert r1 is r2
    k = OrderedDict()
    r1[k] = 1
    assert k in r1
    del r1[k]
    assert k not in r1

    # (4) `last` binds by keyword (unwrap_spec(last=bool)): moves to the front
    kw = {"a": 1, "b": 2, "c": 3}
    mte(kw, "b", last=False)
    assert list(kw) == ["b", "a", "c"]
    mte(kw, "b", last=True)
    assert list(kw) == ["a", "c", "b"]

    # (5) a read-only mappingproxy is not a dict -> TypeError (not reordered)
    class _NS:
        pass

    proxy = _NS.__dict__
    try:
        rd(proxy)
        raise AssertionError("reversed_dict of a mappingproxy must raise TypeError")
    except TypeError:
        pass
    try:
        mte(proxy, "x", last=False)
        raise AssertionError("move_to_end of a mappingproxy must raise TypeError")
    except TypeError:
        pass

    # (6) a no-op reorder (key already at the target) leaves a live iterator
    # valid (length-only iteration guard, matching PyPy)
    noop = {"p": 1, "q": 2, "r": 3}
    it = iter(noop)
    first = next(it)
    mte(noop, "r", last=True)  # "r" is already last -> no reorder
    assert first == "p" and list(it) == ["q", "r"]

    # (7) typed-strategy dicts (int / bytes keys) reorder correctly, both
    # directions, and reversed_dict enumerates them in reverse
    di = {1: "a", 2: "b", 3: "c"}
    mte(di, 1)
    assert list(di) == [2, 3, 1]
    mte(di, 3, last=False)
    assert list(di) == [3, 2, 1]
    assert list(rd(di)) == [1, 2, 3]
    db = {b"x": 1, b"y": 2, b"z": 3}
    mte(db, b"x")
    assert list(db) == [b"y", b"z", b"x"]
    mte(db, b"z", last=False)
    assert list(db) == [b"z", b"y", b"x"]

print("OK")
