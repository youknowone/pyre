"""Python 3.14 memoryview additions layered on PyPy's W_MemoryView typedef."""

import array
import weakref


required = {
    "__buffer__",
    "__class_getitem__",
    "__doc__",
    "__enter__",
    "__eq__",
    "__ge__",
    "__gt__",
    "__exit__",
    "__getitem__",
    "__hash__",
    "__iter__",
    "__len__",
    "__le__",
    "__lt__",
    "__new__",
    "__release_buffer__",
    "__repr__",
    "__setitem__",
    "_from_flags",
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

left = memoryview(array.array("i", [1]))
right = memoryview(array.array("f", [1.0]))
assert left == right
assert left != memoryview(array.array("f", [2.0]))
assert memoryview(array.array("i", [1065353216])) != right
for name in ("__lt__", "__le__", "__gt__", "__ge__"):
    method = memoryview.__dict__[name]
    assert method(left, right) is NotImplemented
    try:
        method(object(), right)
    except TypeError:
        pass
    else:
        raise AssertionError(f"{name} must validate its receiver")
for name in ("__eq__", "__ne__"):
    try:
        memoryview.__dict__[name](object(), right)
    except TypeError:
        pass
    else:
        raise AssertionError(f"{name} must validate its receiver")

flagged = memoryview._from_flags(b"abc", 0)
assert flagged.tobytes() == b"abc" and flagged.readonly
flagged.release()
try:
    memoryview._from_flags(b"abc", 1)
except BufferError as exc:
    assert str(exc) == "Object is not writable."
else:
    raise AssertionError("PyBUF_WRITABLE must reject bytes")
flagged = memoryview._from_flags(bytearray(b"abc"), 1)
assert not flagged.readonly
flagged.release()
source = memoryview(b"abc")
flagged = memoryview._from_flags(source, 1)
assert flagged.readonly and flagged.tobytes() == b"abc"
flagged.release()
source.release()
try:
    memoryview._from_flags(b"abc", 1 << 40)
except OverflowError:
    pass
else:
    raise AssertionError("flags must fit a C int")

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
