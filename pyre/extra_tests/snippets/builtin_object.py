class MyObject:
    pass


assert not MyObject() == MyObject()
assert MyObject() != MyObject()
myobj = MyObject()
assert myobj == myobj
assert not myobj != myobj

object.__subclasshook__(1) == NotImplemented


def assert_type_error(func, *args):
    try:
        func(*args)
    except TypeError:
        pass
    else:
        raise AssertionError("TypeError expected")


# PyPy's interp2app gateway enforces these descriptor signatures.  The Rust
# builtin arity is a dispatch hint, so the implementations must preserve the
# same checks explicitly rather than indexing a missing receiver.
for method in (
    object.__repr__,
    object.__str__,
    object.__format__,
    object.__reduce__,
    object.__reduce_ex__,
    object.__getstate__,
    object.__dir__,
    object.__sizeof__,
):
    assert_type_error(method)

assert_type_error(object.__format__, object())
assert_type_error(object.__reduce__, object(), None)
assert_type_error(object.__getstate__, object(), None)
assert_type_error(object.__dir__, object(), None)
assert_type_error(object.__sizeof__, object(), None)

assert MyObject().__eq__(MyObject()) == NotImplemented
assert MyObject().__ne__(MyObject()) == NotImplemented
assert MyObject().__lt__(MyObject()) == NotImplemented
assert MyObject().__le__(MyObject()) == NotImplemented
assert MyObject().__gt__(MyObject()) == NotImplemented
assert MyObject().__ge__(MyObject()) == NotImplemented

obj = MyObject()

assert obj.__eq__(obj) is True
assert obj.__ne__(obj) is False

assert not hasattr(obj, "a")
obj.__dict__ = {"a": 1}
assert obj.a == 1

del obj.__dict__
d = obj.__dict__
assert isinstance(d, dict)
assert len(d) == 0

try:
    obj.a
    assert False, "AttributeError expected"
except AttributeError:
    pass

# Value inside the formatter goes through a different path of resolution.
# Check that it still works all the same
d = {
    0: "ab",
}
assert "ab ab" == "{k[0]} {vv}".format(k=d, vv=d[0])
