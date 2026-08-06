# pyre-check: platforms=linux,darwin
# The lone surrogate is produced by compiling with a `b"\xff.py"` filename,
# which only decodes where the filesystem encoding is UTF-8 with
# `surrogateescape`.
"""A container's repr concatenates its items' reprs as encoded bytes.

`listobject.py:206-225 _listrepr_inner` builds with a `rutf8.Utf8StringBuilder`
and appends `space.utf8_len_w(space.repr(w_item))`, so an item repr carrying a
lone surrogate — anything a surrogateescape decode produced — reaches the
enclosing repr as itself.  A buffer that can only hold well-formed UTF-8 would
replace it with U+FFFD, which is both a different string and one that no longer
round-trips back to the bytes it came from.
"""

import collections


class C:
    def __repr__(self):
        return "s\udcff"


assert repr(C()) == "s\udcff", ascii(repr(C()))

# every container that re-enters repr per item
assert repr([C()]) == "[s\udcff]", ascii(repr([C()]))
assert repr((C(),)) == "(s\udcff,)", ascii(repr((C(),)))
assert repr({1: C()}) == "{1: s\udcff}", ascii(repr({1: C()}))
assert repr({C()}) == "{s\udcff}", ascii(repr({C()}))
assert repr(frozenset({C()})) == "frozenset({s\udcff})", ascii(repr(frozenset({C()})))
assert repr(collections.deque([C()])) == "deque([s\udcff])", ascii(
    repr(collections.deque([C()]))
)

# and nesting, where the inner repr is itself a built buffer
assert repr([[C()]]) == "[[s\udcff]]", ascii(repr([[C()]]))
assert repr({1: [C()]}) == "{1: [s\udcff]}", ascii(repr({1: [C()]}))

# `str()` of a container formats its items with repr, so it takes the same path
assert str([C()]) == "[s\udcff]", ascii(str([C()]))

# the conversion seams that spell a value with repr
assert f"{C()!r}" == "s\udcff", ascii(f"{C()!r}")
assert "{!r}".format(C()) == "s\udcff", ascii("{!r}".format(C()))
assert "%r" % (C(),) == "s\udcff", ascii("%r" % (C(),))

# an exception argument reaches repr through the args tuple
try:
    raise ValueError(C())
except ValueError as exc:
    assert repr(exc.args) == "(s\udcff,)", ascii(repr(exc.args))

# a code object names a file that never had a UTF-8 spelling, and stays exact
# when a container formats it
code = compile("x = 1", b"\xff.py", "exec")
assert repr(code).count("\udcff") == 1, ascii(repr(code))
assert repr([code]).count("\udcff") == 1, ascii(repr([code]))

# `listobject.py:186-204` keeps the recursion guard, so a container reachable
# from itself still renders the inner reference as an ellipsis rather than
# recursing forever.
recursive_list = []
recursive_list.append(recursive_list)
assert repr(recursive_list) == "[[...]]", ascii(repr(recursive_list))

recursive_dict = {}
recursive_dict[1] = recursive_dict
assert repr(recursive_dict) == "{1: {...}}", ascii(repr(recursive_dict))

# and the depth check still fires, since the items are walked in the host
# language with no Python frame pushed to trip a frame-level limit.
deep = []
for _ in range(200000):
    deep = [deep]
try:
    repr(deep)
except RecursionError:
    pass
else:
    raise AssertionError("a 200000-deep list rendered without raising")

print("OK")
