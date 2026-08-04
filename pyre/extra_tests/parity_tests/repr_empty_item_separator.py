"""A container's repr separates its items by position, not by what they wrote.

`listobject.py:213-223` appends `', '` before every item after the first, and
`tupleobject.py:114` / `dictmultiobject.py:388` join the rendered items the same
way.  An item whose `__repr__` answers `''` therefore still occupies a slot, and
`[a, b]` renders as `'[, ]'` rather than collapsing to `'[]'` — which would be
the repr of a different, empty container.
"""


class Empty:
    def __repr__(self):
        return ""


class EmptyKey(Empty):
    def __hash__(self):
        return 1

    def __eq__(self, other):
        return self is other


import collections

assert repr([Empty(), Empty()]) == "[, ]", ascii(repr([Empty(), Empty()]))
assert repr((Empty(), Empty())) == "(, )", ascii(repr((Empty(), Empty())))
assert repr({Empty(), Empty()}) == "{, }", ascii(repr({Empty(), Empty()}))
assert repr(collections.deque([Empty(), Empty()])) == "deque([, ])", ascii(
    repr(collections.deque([Empty(), Empty()]))
)

# `tupleobject.py:111-113` keeps the one-item form, whose comma is the trailing
# one rather than a separator.
assert repr((Empty(),)) == "(,)", ascii(repr((Empty(),)))

# both halves of a dict entry
assert repr({EmptyKey(): 1, EmptyKey(): 2}) == "{: 1, : 2}", ascii(
    repr({EmptyKey(): 1, EmptyKey(): 2})
)
assert repr({1: Empty(), 2: Empty()}) == "{1: , 2: }", ascii(
    repr({1: Empty(), 2: Empty()})
)
assert repr({EmptyKey(): Empty()}) == "{: }", ascii(repr({EmptyKey(): Empty()}))

# `interp_exceptions.py:135-147 descr_repr` spells the args with
# `repr(tuple(args))`, so it separates by position too.
try:
    raise ValueError(Empty(), Empty())
except ValueError as exc:
    assert repr(exc) == "ValueError(, )", ascii(repr(exc))

# and the same holds one level down, where the rendered item is itself empty
# only because the container it names is
assert repr([[], []]) == "[[], []]", ascii(repr([[], []]))

print("OK")
