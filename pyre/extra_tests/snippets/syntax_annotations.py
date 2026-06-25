from typing import get_type_hints

def func(s: str) -> int:
    return int(s)

hints = get_type_hints(func)

# The order of type hints matters for certain functions
# e.g. functools.singledispatch
assert list(hints.items()) == [('s', str), ('return', int)]


# A module-level annotation makes the compiler emit an implicit
# `__conditional_annotations__` cell (MAKE_CELL) whose binding is written into
# the namespace dict by STORE_NAME. Materializing the namespace mapping with
# `locals()` runs the fast2locals sync, which must leave that binding intact,
# or a later annotation's load of the cell raises NameError.
count: int = 1
assert count == 1

_ = locals()  # fast2locals sync must not erase the implicit cell's binding

maybe: "int | None" = None
assert maybe is None

first: int = 10
second: str = "s"
assert (first, second) == (10, "s")
