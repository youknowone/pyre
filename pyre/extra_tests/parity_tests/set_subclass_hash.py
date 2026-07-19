"""A set subclass may replace the inherited ``__hash__ = None`` slot."""


class HashableSet(set):
    def __hash__(self):
        return int(id(self) & 0x7FFFFFFF)


item = HashableSet()
outer = set()
outer.add(item)
assert item in outer
outer.remove(item)
outer.add(item)
outer.discard(item)

# The strict mapping-key path must use the same subclass method lookup.
d = {item: "value"}
assert d[item] == "value"

try:
    [] in outer
except TypeError as exc:
    # PyPy raises the bare `unhashable type: 'list'`; CPython 3.14 prefixes
    # the element type. Both carry the substring, so match on it.
    assert "unhashable type: 'list'" in str(exc)
else:
    raise AssertionError("unhashable set lookup did not raise")

print("OK")
