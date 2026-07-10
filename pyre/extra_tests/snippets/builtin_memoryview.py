import array

from testutils import assert_raises

obj = b"abcde"
a = memoryview(obj)
assert a.obj == obj

assert a[2:3] == b"c"

assert hash(obj) == hash(a)


class A(array.array): ...


class B(bytes): ...


class C: ...


memoryview(bytearray("abcde", encoding="utf-8"))
memoryview(array.array("i", [1, 2, 3]))
memoryview(A("b", [0]))
memoryview(B("abcde", encoding="utf-8"))

assert_raises(TypeError, lambda: memoryview([1, 2, 3]))
assert_raises(TypeError, lambda: memoryview((1, 2, 3)))
assert_raises(TypeError, lambda: memoryview({}))
assert_raises(TypeError, lambda: memoryview("string"))
assert_raises(TypeError, lambda: memoryview(C()))


def test_slice():
    b = b"123456789"
    m = memoryview(b)
    m2 = memoryview(b)
    assert m == m
    assert m == m2
    assert m.tobytes() == b"123456789"
    assert m == b
    assert m[::2].tobytes() == b"13579"
    assert m[::2] == b"13579"
    assert m[1::2].tobytes() == b"2468"
    assert m[::2][1:].tobytes() == b"3579"
    assert m[::2][1:-1].tobytes() == b"357"
    assert m[::2][::2].tobytes() == b"159"
    assert m[::2][1::2].tobytes() == b"37"
    assert m[::-1].tobytes() == b"987654321"
    assert m[::-2].tobytes() == b"97531"


test_slice()


def test_compare_buffer_exporters():
    # A memoryview compares equal to any operand exporting the same
    # contiguous bytes, not just another memoryview or a bytes-like object:
    # array.array is a non-bytes contiguous exporter.
    m = memoryview(b"abc")
    assert m == array.array("b", [97, 98, 99])
    assert not (m != array.array("b", [97, 98, 99]))
    assert m != array.array("b", [97, 98, 100])
    assert m == memoryview(array.array("b", [97, 98, 99]))
    # A non-buffer operand yields NotImplemented, so the comparison falls
    # through to identity (never equal, never raises).
    assert m != 123
    assert m != "abc"


test_compare_buffer_exporters()


def test_resizable():
    # A live buffer export (memoryview) locks the bytearray against every
    # size-changing mutation; in-place item writes stay legal.  Releasing the
    # export lifts the lock.
    b = bytearray(b"123")
    b.append(4)  # no export yet: legal
    m = memoryview(b)
    assert_raises(BufferError, lambda: b.append(5))
    assert_raises(BufferError, lambda: b.extend(b"xy"))
    assert_raises(BufferError, lambda: b.insert(0, 1))
    assert_raises(BufferError, lambda: b.pop())
    assert_raises(BufferError, lambda: b.remove(ord("1")))
    assert_raises(BufferError, lambda: b.clear())
    assert_raises(BufferError, lambda: b.__iadd__(b"z"))
    assert_raises(BufferError, lambda: b.__setitem__(slice(0, 1), b"ZZZ"))
    assert_raises(BufferError, lambda: b.__delitem__(0))
    assert_raises(BufferError, lambda: b.__delitem__(slice(0, 1)))
    # In-place, size-preserving writes remain legal while exported.
    b[0] = ord("9")
    b[0:2] = b"88"
    assert b[:2] == b"88"
    m.release()
    # Export released: size-changing mutation succeeds again.
    b.append(5)
    assert len(b) == 5
    # A context-managed view re-locks for its lifetime.
    with memoryview(b):
        assert_raises(BufferError, lambda: b.append(6))
    b.append(6)
    assert len(b) == 6


test_resizable()


def test_delitem():
    a = b"abc"
    b = memoryview(a)
    assert_raises(TypeError, lambda: b.__delitem__())
    assert_raises(TypeError, lambda: b.__delitem__(0))
    assert_raises(TypeError, lambda: b.__delitem__(10))
    a = bytearray(b"abc")
    b = memoryview(a)
    assert_raises(TypeError, lambda: b.__delitem__())
    assert_raises(TypeError, lambda: b.__delitem__(1))
    assert_raises(TypeError, lambda: b.__delitem__(12))


test_delitem()
