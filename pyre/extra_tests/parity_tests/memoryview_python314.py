"""Python 3.14 memoryview additions layered on PyPy's W_MemoryView typedef."""

import weakref


required = {
    "__buffer__",
    "__class_getitem__",
    "__doc__",
    "__enter__",
    "__eq__",
    "__exit__",
    "__getitem__",
    "__hash__",
    "__iter__",
    "__len__",
    "__new__",
    "__release_buffer__",
    "__repr__",
    "__setitem__",
    "cast",
    "count",
    "hex",
    "index",
    "release",
    "tobytes",
    "tolist",
    "toreadonly",
}
assert required <= set(memoryview.__dict__)

mv = memoryview(bytearray(b"abaca"))
assert mv.count(ord("a")) == 3
assert mv.count(ord("z")) == 0
assert mv.index(ord("a")) == 0
assert mv.index(ord("a"), 1) == 2
assert mv.index(ord("a"), -2) == 4
assert mv.index(ord("a"), 1, 4) == 2
try:
    mv.index(ord("z"))
except ValueError as exc:
    assert str(exc) == "memoryview.index(x): x not found"
else:
    raise AssertionError("missing memoryview.index value must fail")

exported = mv.__buffer__(0)
assert isinstance(exported, memoryview)
assert exported.tobytes() == mv.tobytes()
exported.release()

alias = memoryview[int]
assert alias.__origin__ is memoryview and alias.__args__ == (int,)

ref = weakref.ref(mv)
assert ref() is mv
mv.release()
for method, args in ((mv.count, (97,)), (mv.index, (97,))):
    try:
        method(*args)
    except ValueError:
        pass
    else:
        raise AssertionError("released memoryview lookup must fail")

print("OK")
