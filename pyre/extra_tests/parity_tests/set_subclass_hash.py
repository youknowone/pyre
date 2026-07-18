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
    assert str(exc) == "cannot use 'list' as a set element (unhashable type: 'list')"
else:
    raise AssertionError("unhashable set lookup did not raise")

print("OK")
