"""Cell TypeDef parity via ``types.CellType`` and ``__closure__``.

PyPy `nestedscope.py:22-125 Cell` + `typedef.py:934-952 Cell.typedef`:

    Cell.typedef = TypeDef("cell",
        ...
        cell_contents = GetSetProperty(
            Cell.descr__cell_contents,
            Cell.descr_set_cell_contents,
            Cell.descr_del_cell_contents,
            cls=Cell),
    )

`descr__cell_contents` returns the inner value or raises
`ValueError("Cell is empty")` when unset (`nestedscope.py:112-116`).
`descr_set_cell_contents` writes the value (`:118-119`).
`descr_del_cell_contents` clears to empty and silently swallows the
`ValueError` for a re-delete (`:121-125`).

Pinned contract:
  1. the Python 3.14 CellType surface is complete,
  2. construction, comparison, repr and hash semantics match 3.14,
  3. `f.__closure__[i]` returns a `cell` (not the unwrapped value),
  4. `cell.cell_contents` reads the captured value,
  5. assignment to `cell.cell_contents` writes through the cell so
     the inner function observes the new value,
  6. `del cell.cell_contents` clears the cell; subsequent reads raise
     `ValueError`.
"""

import types


CellType = types.CellType
assert set(CellType.__dict__) == {
    "__doc__",
    "__eq__",
    "__ge__",
    "__gt__",
    "__hash__",
    "__le__",
    "__lt__",
    "__ne__",
    "__new__",
    "__repr__",
    "cell_contents",
}
assert CellType.__hash__ is None

empty = CellType()
one = CellType(1)
one_again = CellType(1)
two = CellType(2)
assert repr(empty).startswith("<cell at 0x") and repr(empty).endswith(": empty>")
assert repr(one).startswith("<cell at 0x") and ": int object at 0x" in repr(one)
assert empty < one < two
assert empty <= CellType()
assert one == one_again and one != two
assert two > one and two >= one
assert CellType.__eq__(one, 1) is NotImplemented
assert CellType.__lt__(one, 1) is NotImplemented

try:
    hash(one)
except TypeError:
    pass
else:
    assert False, "cell must be unhashable"

try:
    CellType(contents=1)
except TypeError:
    pass
else:
    assert False, "cell constructor arguments are positional-only"

try:
    CellType(1, 2)
except TypeError:
    pass
else:
    assert False, "cell accepts at most one content argument"

try:
    type("CellSubclass", (CellType,), {})
except TypeError:
    pass
else:
    assert False, "cell is not an acceptable base type"

# A cell is an ordinary visible Python object. Storing one in another cell
# must not transparently unwrap the inner cell at an object-space boundary.
nested = CellType(one)
assert nested.cell_contents is one
assert nested != one
assert bool(empty) and bool(one)
assert not isinstance(one, int)
box = [None]
box[0] = one
assert box[0] is one
mapping = {"cell": one}
assert mapping["cell"] is one
try:
    one + 1
except TypeError:
    pass
else:
    raise AssertionError("cell arithmetic must not operate on its contents")


def _make(x):
    def _inner():
        return x
    return _inner

i = _make(10)

# (1) cell identity at the closure tuple position.
c = i.__closure__[0]
assert type(c).__name__ == "cell", f"type(c): {type(c).__name__!r}"

# (2) read cell_contents.
assert c.cell_contents == 10, f"initial cell_contents: {c.cell_contents!r}"

# (3) writing cell_contents propagates to the captured closure.
c.cell_contents = 99
assert c.cell_contents == 99
assert i() == 99, f"inner() after rebind: {i()!r}"

# (4) deleting cell_contents clears the cell; re-read → ValueError.
del c.cell_contents
try:
    _ = c.cell_contents
except ValueError:
    pass
else:
    assert False, "reading cleared cell must raise ValueError"

# Repeating the del is silently ignored per nestedscope.py:121-125.
del c.cell_contents
# After a fresh write the cell is alive again.
c.cell_contents = 7
assert c.cell_contents == 7

print("OK")
