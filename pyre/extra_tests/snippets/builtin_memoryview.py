import array
import copy
import gc
import io
import mmap
import os
import pickle
import struct
import tempfile
import weakref

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
assert memoryview(object=b"abcde").tobytes() == b"abcde"

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

    # CPython 3.14 `memory_subscript` registers the result view before slice
    # bound conversion.  Releasing the source from `__index__` therefore
    # rejects scalar access but leaves a slice result backed by the original
    # export (gh-92888).
    source = bytearray(b"abcdefgh")
    m = memoryview(source)

    class ReleasingIndex:
        def __index__(self):
            m.release()
            return 4

    assert_raises(ValueError, lambda: m[ReleasingIndex()])

    source = bytearray(b"abcdefgh")
    m = memoryview(source)
    sliced = m[:ReleasingIndex()]
    assert sliced.tobytes() == b"abcd"
    assert_raises(BufferError, lambda: source.append(0))
    sliced.release()
    source.append(0)


test_slice()


def test_half_float():
    packed = struct.pack("eee", 0.0, -1.5, 1.5)
    view = memoryview(packed).cast("e")
    assert view.itemsize == 2
    assert view.tolist() == [0.0, -1.5, 1.5]


test_half_float()


def test_index_uses_element_shape():
    values = array.array("i", [10, 20, 30])
    view = memoryview(values)
    assert view.index(30) == 2
    assert view.index(20, -3, -1) == 1
    assert_raises(ValueError, lambda: view.index(40))
    assert_raises(ValueError, lambda: view.index(30, 0, 2))


test_index_uses_element_shape()


def test_bytesio_readinto():
    stream = io.BytesIO(b"hello")
    target = bytearray(b"testing")
    assert stream.readinto(target) == 5
    assert target == bytearray(b"hellong")

    stream.seek(0)
    writable = memoryview(bytearray(5))
    assert stream.readinto1(writable) == 5
    assert writable.tobytes() == b"hello"

    stream.seek(0)
    assert_raises(TypeError, lambda: stream.readinto(memoryview(b"hello")))


test_bytesio_readinto()


# PyPy interp_buffer.py descr_new_picklebuffer acquires the buffer in
# __new__, and its typedef is explicitly final.  CPython 3.14 exposes the
# same constructor and final-type behavior through pickle.PickleBuffer.
pickle_buffer_source = bytearray(b"abc")
pickle_buffer = pickle.PickleBuffer(memoryview(pickle_buffer_source))
assert pickle_buffer.raw().tobytes() == b"abc"
pickle_buffer_source[0] = ord("z")
assert pickle_buffer.raw().tobytes() == b"zbc"
assert weakref.ref(pickle_buffer)() is pickle_buffer

assert_raises(TypeError, lambda: pickle.PickleBuffer(1))
released_pickle_buffer_source = memoryview(b"released")
released_pickle_buffer_source.release()
assert_raises(ValueError, lambda: pickle.PickleBuffer(released_pickle_buffer_source))


def subclass_pickle_buffer():
    class PickleBufferSubclass(pickle.PickleBuffer):
        pass


assert_raises(TypeError, subclass_pickle_buffer)


def test_pickle_rejected():
    view = memoryview(b"abc")
    assert_raises(TypeError, lambda: copy.copy(view))
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        assert_raises(TypeError, lambda protocol=protocol: pickle.dumps(view, protocol))


test_pickle_rejected()


def test_reentrant_release_is_export_guarded():
    class HashingArray(array.array):
        def __hash__(self):
            hash_view.release()
            self.clear()
            return 123

    exporter = HashingArray("B", b"A" * 32)
    hash_view = memoryview(exporter).toreadonly()
    assert_raises(BufferError, lambda: hash(hash_view))

    backing = bytearray(b"A" * 32)
    hex_view = memoryview(backing)

    class Separator(bytes):
        def __len__(self):
            hex_view.release()
            backing.clear()
            return 1

    assert_raises(BufferError, lambda: hex_view.hex(Separator(b":")))


test_reentrant_release_is_export_guarded()


def test_weakref_and_exporter_cycles():
    callbacks = []
    view = memoryview(b"abc")
    ref = weakref.ref(view, lambda dead: callbacks.append(dead))
    del view
    gc.collect()
    assert ref() is None
    assert callbacks == [ref]

    class Source(bytearray):
        pass

    class Payload:
        pass

    source = Source(b"abc")
    view = memoryview(source)
    payload = Payload()
    source.view = view
    source.payload = payload
    payload_ref = weakref.ref(payload)
    del source, view, payload
    gc.collect()
    assert payload_ref() is None


test_weakref_and_exporter_cycles()


def test_mmap_buffer_protocol():
    fd, path = tempfile.mkstemp()
    try:
        os.ftruncate(fd, 64)
        mapped = mmap.mmap(fd, 64)
        view = memoryview(mapped)
        struct.pack_into("q", view, 0, 1234)
        assert struct.unpack_from("q", view, 0) == (1234,)
        view.release()
        mapped.close()

        with open(path, "wb") as output:
            assert output.write(memoryview(b"abc")) == 3
        with open(path, "rb") as source:
            assert source.read() == b"abc"
    finally:
        os.close(fd)
        os.unlink(path)


test_mmap_buffer_protocol()


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
