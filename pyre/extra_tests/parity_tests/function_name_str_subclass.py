"""Phase 6 parity test: __name__ / __qualname__ setters accept str subclasses.

PyPy `function.py:462-468 fset_func_name`:

    def fset_func_name(self, space, w_name):
        self._check_code_mutable("__name__")
        if space.isinstance_w(w_name, space.w_text):
            self.name = space.text_w(w_name)
        else:
            raise oefmt(space.w_TypeError,
                        "__name__ must be set to a string object")

PyPy `function.py:476-485 fset_func_qualname`:

    def fset_func_qualname(self, space, w_name):
        self._check_code_mutable("__qualname__")
        try:
            qualname = space.realutf8_w(w_name)
        except OperationError as e:
            if e.match(space, space.w_TypeError):
                raise oefmt(space.w_TypeError,
                            "__qualname__ must be set to a string object")
            raise
        self.set_qualname(qualname)

Both use `isinstance_w(w_name, w_text)` or `realutf8_w` (which itself
falls through to the same `isinstance_w` check), so a `str` subclass
should be accepted.

Pinned contract:
  1. Setting __name__ to a plain str works (baseline).
  2. Setting __name__ to a str subclass instance works (PyPy parity).
  3. Setting __name__ to a non-str (int, None) raises TypeError.
  4. Same three cases for __qualname__.
"""

class MyStr(str):
    pass

def _f():
    return 0

# (1) Plain str.
_f.__name__ = "renamed"
assert _f.__name__ == "renamed"

# (2) str subclass.
_f.__name__ = MyStr("subname")
assert _f.__name__ == "subname", f"after subclass: {_f.__name__!r}"

# (3) Non-str rejected.
for bad in (1, None, [1], (1,), object()):
    try:
        _f.__name__ = bad
    except TypeError:
        pass
    else:
        assert False, f"setting __name__ to {bad!r} must raise TypeError"

# (4) __qualname__ same protocol.
qualname = "".join(("qual",))
_f.__qualname__ = qualname
assert _f.__qualname__ == "qual"
assert _f.__qualname__ is qualname

subqualname = MyStr("subqual")
_f.__qualname__ = subqualname
assert _f.__qualname__ == "subqual"
assert _f.__qualname__ is subqualname

for bad in (1, None, [1], (1,), object()):
    try:
        _f.__qualname__ = bad
    except TypeError:
        pass
    else:
        assert False, f"setting __qualname__ to {bad!r} must raise TypeError"

print("OK")
