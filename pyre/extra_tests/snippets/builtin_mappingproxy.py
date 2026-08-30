from testutils import assert_raises
from types import MappingProxyType


class A(dict):
    def a():
        pass

    def b():
        pass


assert A.__dict__["a"] == A.a
with assert_raises(KeyError) as cm:
    A.__dict__["not here"]

assert cm.exception.args[0] == "not here"

assert "b" in A.__dict__
assert "c" not in A.__dict__

assert "__dict__" in A.__dict__

assert A.__dict__.get("not here", "default") == "default"
assert A.__dict__.get("a", "default") is A.a
assert A.__dict__.get("not here") is None


# PyPy `W_DictProxyObject.descr_reversed` delegates to the wrapped mapping;
# mappingproxy accepts arbitrary mappings, not only dict-backed objects.
class CustomMapping:
    def __getitem__(self, key):
        raise KeyError(key)

    def __reversed__(self):
        return iter(("custom-reversed",))


assert list(reversed(MappingProxyType(CustomMapping()))) == ["custom-reversed"]
with assert_raises(TypeError):
    MappingProxyType.get({}, "key")
with assert_raises(TypeError):
    MappingProxyType(CustomMapping()).__reversed__("extra")


# PyPy `W_DictProxyObject.descr_ror` invokes the left dict's `__or__` after
# unwrapping the proxy receiver. Preserve an overriding dict subclass method.
or_calls = []


class CustomOrDict(dict):
    def __or__(self, other):
        or_calls.append(other)
        return "custom-or"


assert MappingProxyType({"left": 1}).__or__(CustomOrDict(right=2)) == {
    "left": 1,
    "right": 2,
}
assert or_calls == []

right = {"right": 2}
assert MappingProxyType(right).__ror__(CustomOrDict(left=1)) == "custom-or"
assert or_calls == [right]
