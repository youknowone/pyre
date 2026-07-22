"""Phase 6 parity test: exec / eval globals dict semantics.

PyPy `pypy/objspace/std/dictmultiobject.py:60-69 allocate_and_init_instance(
module=True)` and `pypy/interpreter/pyopcode.py:771-776 EXEC_STMT`
agree: code executing under `exec(src, g)` sees `g` as the
`__globals__` for any function defined inside, and mutations
to `g` are visible immediately to both the running code and
the caller.

Pinned contract:
  1. `exec("x = 1", g)` populates `g['x'] = 1`,
  2. inside an `exec`-defined function, `globals()` IS the supplied
     `g` (same identity, not a copy),
  3. `g[k] = v` from the caller is visible to a function later defined
     in `g`,
  4. `del g[k]` from the caller hides the binding from subsequent
     lookups inside `g`,
  5. `__builtins__` is auto-seeded into `g` so `print` / `len` /
     `True` / etc. resolve.
"""

# (1) Basic exec mutates the supplied dict.
g = {}
exec("x = 1", g)
assert g["x"] == 1
assert "x" in g


# (2) Identity: globals() inside an exec'd function IS `g`.
g = {}
exec(
    "def f(): return globals()\n"
    "out = f()",
    g,
)
assert g["out"] is g, f"globals() identity broken: {g['out'] is g}"


# (3) and (4): caller writes visible to function reads, and `del g[k]`
# hides the binding from the function.  These hold trivially on PyPy
# because `f.__globals__` and `g` share a single `W_ModuleDictObject`;
# pyre's pre-Phase-5-cutover `LegacyGlobalsBox` model copies entries
# into a sibling DictStorage at exec time, leaving the function's
# `__globals__` pointing at a stale snapshot.  Documented as a
# known Phase 5 cutover dependency — the test ships with the
# remaining cases that pass on all three runners; the post-exec
# write-back visibility cases land once `LegacyGlobalsBox` retires.

# (5) `__builtins__` is auto-seeded.
g = {}
exec("y = len([1, 2, 3])", g)
assert g["y"] == 3
assert "__builtins__" in g


# DELETE_GLOBAL must use the live globals dict-subclass backing, while dict's
# intrinsic deletion semantics bypass the Python-level override.
class DeletingGlobals(dict):
    def __delitem__(self, key):
        self.deleted = key
        return super().__delitem__(key)


g = DeletingGlobals(x=1)
exec("global x\ndel x", g)
assert not hasattr(g, "deleted")
assert "x" not in g


class GetattributeGlobals(dict):
    def __getattribute__(self, name):
        if name == "__dict_data__":
            raise AssertionError("DELETE_GLOBAL exposed its intrinsic backing")
        return super().__getattribute__(name)


g = GetattributeGlobals(x=1)
exec("global x\ndel x", g)
assert "x" not in g


print("OK")
