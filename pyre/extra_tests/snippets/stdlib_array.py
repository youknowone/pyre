import gc
from array import _array_reconstructor, array
from io import BytesIO
from pickle import dumps, loads

from testutils import assert_raises

a1 = array("b", [0, 1, 2, 3])

assert a1.tobytes() == b"\x00\x01\x02\x03"
assert a1[2] == 2

assert list(a1) == [0, 1, 2, 3]

a1.reverse()
assert a1 == array("B", [3, 2, 1, 0])

a1.extend([4, 5, 6, 7])

assert a1 == array("h", [3, 2, 1, 0, 4, 5, 6, 7])

# eq, ne
a = array("b", [0, 1, 2, 3])
b = a
assert a.__ne__(b) is False
b = array("B", [3, 2, 1, 0])
assert a.__ne__(b) is True


def test_float_with_integer_input():
    f = array("f", [0, 1, 2.0, 3.0])
    f.append(4)
    f.insert(0, -1)
    assert f.count(4) == 1
    f.remove(1)
    assert f.index(0) == 1
    f[0] = -2
    assert f == array("f", [-2, 0, 2, 3, 4])


test_float_with_integer_input()

# PyPy's W_Array.descr_append converts before setlen.  CPython 3.14's ins1
# preserves that ordering but validates once before resize and converts again
# for the actual slot, so both callbacks and their intervening mutations are
# observable.
append_target = array("i", [1])


class GrowingAppendItem:
    def __init__(self):
        self.calls = 0

    def __index__(self):
        self.calls += 1
        gc.collect()
        append_target.append(2)
        return 3


growing_append_item = GrowingAppendItem()
append_target.append(growing_append_item)
assert growing_append_item.calls == 2
assert append_target == array("i", [1, 3, 2])

append_target = array("i", [1])


class ClearingAppendItem:
    def __init__(self):
        self.calls = 0

    def __index__(self):
        self.calls += 1
        del append_target[:]
        return 3


clearing_append_item = ClearingAppendItem()
append_target.append(clearing_append_item)
assert clearing_append_item.calls == 2
assert append_target == array("i")

append_target = array("i", [1])
append_view = memoryview(append_target)


class ExportedAppendItem:
    def __init__(self):
        self.calls = 0

    def __index__(self):
        self.calls += 1
        return 3


exported_append_item = ExportedAppendItem()
with assert_raises(BufferError):
    append_target.append(exported_append_item)
assert exported_append_item.calls == 1
assert append_target == array("i", [1])
append_view.release()


class AppendArraySubclass(array):
    pass


with assert_raises(TypeError) as error:
    AppendArraySubclass("i").append(v=1)
assert str(error.exception) == "array.append() takes no keyword arguments"
with assert_raises(TypeError) as error:
    array("i").append()
assert str(error.exception) == "array.append() takes exactly one argument (0 given)"
with assert_raises(TypeError) as error:
    array("i").append(1, 2)
assert str(error.exception) == "array.append() takes exactly one argument (2 given)"

# PyPy W_Array.extend raw-copies a same-kind array before falling back to
# _fromiterable, and that fallback mints the iterator before its first append.
# CPython 3.14 shares append's two-conversion ins1 path for every streamed item.
extend_target = array("i", [1])


class GrowingExtendItem:
    def __init__(self):
        self.calls = 0

    def __index__(self):
        self.calls += 1
        extend_target.append(2)
        gc.collect()
        return 3


growing_extend_item = GrowingExtendItem()
extend_target.extend([growing_extend_item])
assert growing_extend_item.calls == 2
assert extend_target == array("i", [1, 3, 2])

extend_target = array("i", [1])


class ClearingExtendItem:
    def __init__(self):
        self.calls = 0

    def __index__(self):
        self.calls += 1
        del extend_target[:]
        return 3


clearing_extend_item = ClearingExtendItem()
extend_target.extend([clearing_extend_item])
assert clearing_extend_item.calls == 2
assert extend_target == array("i")


class RawArraySubclass(array):
    def __iter__(self):
        raise AssertionError("same-kind array extend must copy raw storage")


extend_target = array("i", [1])
extend_target.extend(RawArraySubclass("i", [2]))
assert extend_target == array("i", [1, 2])

exported_extend = array("i", [1])
exported_extend_view = memoryview(exported_extend)
with assert_raises(TypeError) as error:
    exported_extend.extend(array("h"))
assert str(error.exception) == "can only extend with array of same kind"
assert exported_extend.extend(array("i")) is None
assert exported_extend.extend([]) is None


class EmptyExtendIterator:
    def __init__(self):
        self.events = []

    def __iter__(self):
        self.events.append("iter")
        return self

    def __next__(self):
        self.events.append("next")
        raise StopIteration


empty_extend_iterator = EmptyExtendIterator()
assert exported_extend.extend(empty_extend_iterator) is None
assert empty_extend_iterator.events == ["iter", "next"]


class ExportedExtendItem:
    def __init__(self):
        self.calls = 0

    def __index__(self):
        self.calls += 1
        return 2


exported_extend_item = ExportedExtendItem()
with assert_raises(BufferError):
    exported_extend.extend([exported_extend_item])
assert exported_extend_item.calls == 1
assert exported_extend == array("i", [1])
exported_extend_view.release()

with assert_raises(TypeError) as error:
    array("i").extend()
assert str(error.exception) == "extend() takes exactly 1 positional argument (0 given)"
with assert_raises(TypeError) as error:
    array("i").extend([], [])
assert str(error.exception) == "extend() takes at most 1 argument (2 given)"
with assert_raises(TypeError) as error:
    RawArraySubclass("i").extend(iterable=[])
assert str(error.exception) == "extend() takes exactly 1 positional argument (0 given)"

# PyPy W_Array.descr_insert converts the value before growing and shifting.
# CPython 3.14's ins1 retains that order, then converts once more for the
# destination slot; both callbacks and the pre-conversion length are visible.
insert_target = array("i", [1, 2])


class GrowingInsertItem:
    def __init__(self):
        self.calls = 0

    def __index__(self):
        self.calls += 1
        if self.calls == 1:
            insert_target.append(9)
        else:
            insert_target.append(8)
        return 7


growing_insert_item = GrowingInsertItem()
insert_target.insert(-1, growing_insert_item)
assert growing_insert_item.calls == 2
assert insert_target == array("i", [1, 7, 2, 8])

insert_target = array("i", [1])
insert_view = None


class ExportingInsertItem:
    def __index__(self):
        global insert_view
        insert_view = memoryview(insert_target)
        return 2


with assert_raises(BufferError):
    insert_target.insert(0, ExportingInsertItem())
assert insert_target == array("i", [1])
insert_view.release()

insert_target = array("i", [1, 2])


class ClearingInsertItem:
    def __init__(self):
        self.calls = 0

    def __index__(self):
        self.calls += 1
        if self.calls == 2:
            insert_target.clear()
        return 7


clearing_insert_item = ClearingInsertItem()
with assert_raises(IndexError) as error:
    insert_target.insert(0, clearing_insert_item)
assert str(error.exception) == "array assignment index out of range"
assert clearing_insert_item.calls == 2
assert insert_target == array("i")

# PyPy's W_ArrayBase.descr_index supplies the optional bounds to
# index_count_array.  CPython 3.14 keeps the same observable positional search
# but makes the public bounds positional-only.
bounded = array("i", [1, 2, 1, 3])
assert bounded.index(1, 1) == 2
assert bounded.index(1, 1, 3) == 2
with assert_raises(TypeError) as error:
    bounded.index(1, start=1)
assert str(error.exception) == "array.index() takes no keyword arguments"
with assert_raises(TypeError) as error:
    bounded.index(x=1)
assert str(error.exception) == "array.index() takes no keyword arguments"


class ArraySubclass(array):
    pass


with assert_raises(TypeError) as error:
    ArraySubclass("i", [1]).index(1, start=0)
assert str(error.exception) == "array.index() takes no keyword arguments"
with assert_raises(TypeError) as error:
    bounded.index()
assert str(error.exception) == "index expected at least 1 argument, got 0"
with assert_raises(TypeError) as error:
    bounded.index(1, 0, 4, 5)
assert str(error.exception) == "index expected at most 3 arguments, got 4"
with assert_raises(TypeError) as error:
    bounded.index(1, None)
assert str(error.exception) == (
    "slice indices must be integers or have an __index__ method"
)
with assert_raises(TypeError) as error:
    bounded.index(1, 0, None)
assert str(error.exception) == (
    "slice indices must be integers or have an __index__ method"
)


# interp2app unwraps bounds before W_ArrayBase.descr_index reads self.len.
reentered = array("i", [1])


class GrowingStart:
    def __index__(self):
        reentered.append(2)
        return -1


assert reentered.index(2, GrowingStart()) == 1
assert reentered == array("i", [1, 2])


class CollectingNeedle:
    def __eq__(self, other):
        gc.collect()
        return other == 2


assert array("i", [1, 2]).index(CollectingNeedle()) == 1

growing_search = array("i", [1])


class GrowingNeedle:
    def __eq__(self, other):
        if other == 1:
            growing_search.append(2)
            return False
        return other == 2


# PyPy's index_count_array source reads arr.len at its loop boundary.  The
# installed PyPy oracle snapshots this case, while CPython 3.14's
# array_array_index_impl explicitly re-reads Py_SIZE(self) and finds the new
# item.  Keep the PyPy loop shape and take the 3.14 observable result.
assert growing_search.index(GrowingNeedle()) == 1

shrinking = array("i", [1, 2])


class ShrinkingNeedle:
    def __eq__(self, other):
        del shrinking[:]
        return False


with assert_raises(ValueError):
    shrinking.index(ShrinkingNeedle())
assert shrinking == array("i")

# PyPy's unwrap_spec gateway converts insert/pop indexes before their method
# bodies inspect the receiver.  The callbacks can therefore resize or collect
# the array before its effective length is read.
inserted = array("i", [1, 2])


class GrowingInsertIndex:
    def __index__(self):
        inserted.append(3)
        gc.collect()
        return -1


inserted.insert(GrowingInsertIndex(), 9)
assert inserted == array("i", [1, 2, 9, 3])

popped = array("i", [1, 2])


class GrowingPopIndex:
    def __index__(self):
        popped.append(3)
        gc.collect()
        return -1


assert popped.pop(GrowingPopIndex()) == 3
assert popped == array("i", [1, 2])

cleared = array("i", [1, 2])


class ClearingPopIndex:
    def __index__(self):
        del cleared[:]
        return 0


with assert_raises(IndexError) as error:
    cleared.pop(ClearingPopIndex())
assert str(error.exception) == "pop from empty array"
assert cleared == array("i")


class HugeArrayIndex:
    def __index__(self):
        return 1 << 100


for method in (
    lambda: array("i").insert(HugeArrayIndex(), 1),
    lambda: array("i", [1]).pop(HugeArrayIndex()),
):
    with assert_raises(OverflowError) as error:
        method()
    assert str(error.exception) == "Python int too large to convert to C ssize_t"

with assert_raises(TypeError) as error:
    array("i").insert(i=0, x=1)
assert str(error.exception) == "array.insert() takes no keyword arguments"
with assert_raises(TypeError) as error:
    array("i", [1]).pop(i=0)
assert str(error.exception) == "array.pop() takes no keyword arguments"
with assert_raises(TypeError) as error:
    array("i").insert(0)
assert str(error.exception) == "insert expected 2 arguments, got 1"
with assert_raises(TypeError) as error:
    array("i", [1]).pop(0, 1)
assert str(error.exception) == "pop expected at most 1 argument, got 2"

conversion_order = []
exported = array("i", [1])
exported_view = memoryview(exported)


class ExportingIndex:
    def __index__(self):
        conversion_order.append("index")
        return 0


class ExportingValue:
    def __index__(self):
        conversion_order.append("value")
        return 2


with assert_raises(BufferError):
    exported.insert(ExportingIndex(), ExportingValue())
assert conversion_order == ["index", "value"]
conversion_order.clear()
with assert_raises(BufferError):
    exported.pop(ExportingIndex())
assert conversion_order == ["index"]
exported_view.release()

# PyPy's descr_remove delegates through its live-length index_count_array
# search and then descr_pop.  Equality can mutate the receiver at both sides
# of that boundary, so the receiver and needle must remain rooted and live.
remove_growing = array("i", [1, 2])
remove_calls = []


class GrowingRemoveNeedle:
    def __eq__(self, other):
        remove_calls.append(other)
        if other == 1:
            remove_growing.append(3)
            gc.collect()
            return False
        return other == 3


remove_growing.remove(GrowingRemoveNeedle())
assert remove_calls == [1, 2, 3]
assert remove_growing == array("i", [1, 2])

remove_cleared_false = array("i", [1, 2])


class ClearingFalseNeedle:
    def __eq__(self, other):
        del remove_cleared_false[:]
        return False


with assert_raises(ValueError) as error:
    remove_cleared_false.remove(ClearingFalseNeedle())
assert str(error.exception) == "array.remove(x): x not in array"
assert remove_cleared_false == array("i")

remove_cleared_true = array("i", [1, 2])


class ClearingTrueNeedle:
    def __eq__(self, other):
        del remove_cleared_true[:]
        return True


# CPython 3.14 array_del_slice treats the now-empty matched range as a
# successful no-op; PyPy's descr_pop raises here, so the 3.14 result wins.
assert remove_cleared_true.remove(ClearingTrueNeedle()) is None
assert remove_cleared_true == array("i")

remove_exported = array("i", [1])
remove_view = memoryview(remove_exported)
remove_export_events = []


class ExportedRemoveNeedle:
    def __eq__(self, other):
        remove_export_events.append(other)
        return True


with assert_raises(BufferError):
    remove_exported.remove(ExportedRemoveNeedle())
assert remove_export_events == [1]
assert remove_exported == array("i", [1])
remove_view.release()

with assert_raises(TypeError) as error:
    ArraySubclass("i", [1]).remove(x=1)
assert str(error.exception) == "array.remove() takes no keyword arguments"
with assert_raises(TypeError) as error:
    array("i").remove()
assert str(error.exception) == "array.remove() takes exactly one argument (0 given)"
with assert_raises(TypeError) as error:
    array("i").remove(1, 2)
assert str(error.exception) == "array.remove() takes exactly one argument (2 given)"

# PyPy's index_count_array shares one live-length loop between its index-like
# searches and count.  CPython 3.14 likewise re-reads Py_SIZE(self) at every
# array_array_count_impl iteration, so comparison may grow or shrink the
# receiver without leaving a stale raw-buffer bound behind.
count_growing = array("i", [1, 2])
count_calls = []


class GrowingCountNeedle:
    def __eq__(self, other):
        count_calls.append(other)
        if other == 1:
            count_growing.append(3)
            gc.collect()
        return other == 3


assert count_growing.count(GrowingCountNeedle()) == 1
assert count_calls == [1, 2, 3]
assert count_growing == array("i", [1, 2, 3])

count_cleared_false = array("i", [1, 2])


class ClearingFalseCountNeedle:
    def __eq__(self, other):
        del count_cleared_false[:]
        return False


assert count_cleared_false.count(ClearingFalseCountNeedle()) == 0
assert count_cleared_false == array("i")

count_cleared_true = array("i", [1, 2])


class ClearingTrueCountNeedle:
    def __eq__(self, other):
        del count_cleared_true[:]
        return True


assert count_cleared_true.count(ClearingTrueCountNeedle()) == 1
assert count_cleared_true == array("i")

with assert_raises(TypeError) as error:
    ArraySubclass("i", [1]).count(x=1)
assert str(error.exception) == "array.count() takes no keyword arguments"
with assert_raises(TypeError) as error:
    array("i").count()
assert str(error.exception) == "array.count() takes exactly one argument (0 given)"
with assert_raises(TypeError) as error:
    array("i").count(1, 2)
assert str(error.exception) == "array.count() takes exactly one argument (2 given)"

# PyPy's W_ArrayBase.descr_fromlist accepts only lists and restores the old
# length around fromsequence failures.  CPython 3.14 additionally reads list
# subclasses directly and rejects a size change after each converted item.
fromlist_target = array("i", [9])
with assert_raises(TypeError) as error:
    fromlist_target.fromlist((1, 2))
assert str(error.exception) == "arg must be list"
assert fromlist_target == array("i", [9])

with assert_raises(TypeError):
    fromlist_target.fromlist([1, "bad", 2])
assert fromlist_target == array("i", [9])


class DirectListSubclass(list):
    def __iter__(self):
        raise AssertionError("fromlist must read list storage directly")


fromlist_target.fromlist(DirectListSubclass([1, 2]))
assert fromlist_target == array("i", [9, 1, 2])

growing_fromlist_source = []


class GrowingFromlistItem:
    def __index__(self):
        growing_fromlist_source.append(2)
        gc.collect()
        return 1


growing_fromlist_source.append(GrowingFromlistItem())
fromlist_target = array("i", [9])
with assert_raises(RuntimeError) as error:
    fromlist_target.fromlist(growing_fromlist_source)
assert str(error.exception) == "list changed size during iteration"
assert fromlist_target == array("i", [9])

shrinking_fromlist_source = []


class ShrinkingFromlistItem:
    def __index__(self):
        shrinking_fromlist_source.clear()
        return 1


shrinking_fromlist_source.append(ShrinkingFromlistItem())
fromlist_target = array("i", [9])
with assert_raises(RuntimeError) as error:
    fromlist_target.fromlist(shrinking_fromlist_source)
assert str(error.exception) == "list changed size during iteration"
assert fromlist_target == array("i", [9])

# Destination sizing precedes item conversion.  Re-entrant growth therefore
# remains after the pre-sized slot, while a clear makes that slot non-live.
reentered_fromlist = array("i", [9])


class AppendingFromlistItem:
    def __index__(self):
        reentered_fromlist.append(7)
        return 1


reentered_fromlist.fromlist([AppendingFromlistItem()])
assert reentered_fromlist == array("i", [9, 1, 7])

reentered_fromlist = array("i", [9])


class ClearingFromlistItem:
    def __index__(self):
        del reentered_fromlist[:]
        return 1


reentered_fromlist.fromlist([ClearingFromlistItem()])
assert reentered_fromlist == array("i")

# Type/empty-list handling happens before a resize check.  A non-empty list is
# the first case that needs to grow the exported array.
exported_fromlist = array("i", [9])
exported_fromlist_view = memoryview(exported_fromlist)
with assert_raises(TypeError) as error:
    exported_fromlist.fromlist(())
assert str(error.exception) == "arg must be list"
assert exported_fromlist.fromlist([]) is None
with assert_raises(BufferError):
    exported_fromlist.fromlist([1])
assert exported_fromlist == array("i", [9])
exported_fromlist_view.release()

with assert_raises(TypeError) as error:
    ArraySubclass("i").fromlist(list=[])
assert str(error.exception) == "array.fromlist() takes no keyword arguments"
with assert_raises(TypeError) as error:
    array("i").fromlist()
assert str(error.exception) == "array.fromlist() takes exactly one argument (0 given)"
with assert_raises(TypeError) as error:
    array("i").fromlist([], [])
assert str(error.exception) == "array.fromlist() takes exactly one argument (2 given)"

# PyPy decode_index4(w_idx, self) and CPython 3.14 array_subscr /
# array_ass_subscr read the receiver length after an index conversion.  The
# callback can resize and collect the receiver, so negative indexes follow the
# new tail on get, set, and delete.
indexed = array("i", [1, 2])


class GrowingSubscriptionIndex:
    def __index__(self):
        indexed.append(3)
        gc.collect()
        return -1


assert indexed[GrowingSubscriptionIndex()] == 3
assert indexed == array("i", [1, 2, 3])

indexed = array("i", [1, 2])
indexed[GrowingSubscriptionIndex()] = 9
assert indexed == array("i", [1, 2, 9])

indexed = array("i", [1, 2])
del indexed[GrowingSubscriptionIndex()]
assert indexed == array("i", [1, 2])

for operation in ("get", "delete"):
    indexed = array("i", [1, 2])

    class ClearingSubscriptionIndex:
        def __index__(self):
            indexed.clear()
            return 0

    with assert_raises(IndexError) as error:
        if operation == "get":
            indexed[ClearingSubscriptionIndex()]
        else:
            del indexed[ClearingSubscriptionIndex()]
    expected = (
        "array index out of range"
        if operation == "get"
        else "array assignment index out of range"
    )
    assert str(error.exception) == expected
    assert indexed == array("i")

# W_SliceObject.unpack likewise runs every bound conversion before
# adjust_indices reads self.len.  A newly appended last item is therefore the
# item selected by a -1 lower bound.
class GrowingSliceStart:
    def __index__(self):
        sliced.append(3)
        gc.collect()
        return -1


sliced = array("i", [1, 2])
assert sliced[GrowingSliceStart() :] == array("i", [3])
assert sliced == array("i", [1, 2, 3])

sliced = array("i", [1, 2])
sliced[GrowingSliceStart() :] = array("i", [9])
assert sliced == array("i", [1, 2, 9])

sliced = array("i", [1, 2])
del sliced[GrowingSliceStart() :]
assert sliced == array("i", [1, 2])

# slice assignment step overflow behaviour test
T = "I"
a = array(T, range(10))
b = array(T, [100])
a[::9999999999] = b
assert a == array(T, [100, 1, 2, 3, 4, 5, 6, 7, 8, 9])
a[::-9999999999] = b
assert a == array(T, [100, 1, 2, 3, 4, 5, 6, 7, 8, 100])
c = array(T)
a[0:0:9999999999] = c
assert a == array(T, [100, 1, 2, 3, 4, 5, 6, 7, 8, 100])
a[0:0:-9999999999] = c
assert a == array(T, [100, 1, 2, 3, 4, 5, 6, 7, 8, 100])
del a[::9999999999]
assert a == array(T, [1, 2, 3, 4, 5, 6, 7, 8, 100])
del a[::-9999999999]
assert a == array(T, [1, 2, 3, 4, 5, 6, 7, 8])
del a[0:0:9999999999]
assert a == array(T, [1, 2, 3, 4, 5, 6, 7, 8])
del a[0:0:-9999999999]
assert a == array(T, [1, 2, 3, 4, 5, 6, 7, 8])


def test_float_with_nan():
    f = float("nan")
    a = array("f")
    a.append(f)
    assert not (a == a)
    assert a != a
    assert not (a < a)
    assert not (a <= a)
    assert not (a > a)
    assert not (a >= a)


test_float_with_nan()


def test_different_type_cmp():
    a = array("i", [-1, -2, -3, -4])
    b = array("I", [1, 2, 3, 4])
    c = array("f", [1, 2, 3, 4])
    assert a < b
    assert b > a
    assert b == c
    assert a < c
    assert c > a


test_different_type_cmp()


def test_array_frombytes():
    a = array("b", [-1, -2])
    b = bytearray(a.tobytes())
    c = array("b", b)
    assert a == c


test_array_frombytes()

# test that indexing on an empty array doesn't panic
a = array("b")
with assert_raises(IndexError):
    a[0]
with assert_raises(IndexError):
    a[0] = 42
with assert_raises(IndexError):
    del a[42]

test_str = "🌉abc🌐def🌉🌐"
u = array("u", test_str)
# skip as 2 bytes character environment with CPython is failing the test
if u.itemsize >= 4:
    assert u.__reduce_ex__(1)[1][1] == list(test_str)
    assert loads(dumps(u, 1)) == loads(dumps(u, 3))

# test array name
a = array("b", [])
assert str(a.__class__.__name__) == "array"
# test arrayiterator name
i = iter(a)
assert str(i.__class__.__name__) == "arrayiterator"

# teset array.__contains__
a = array("B", [0])
assert a.__contains__(0)
assert not a.__contains__(1)


class _ReenteringWriter:
    def __init__(self, arr):
        self.arr = arr
        self.reentered = False

    def write(self, chunk):
        if not self.reentered:
            self.reentered = True
            self.arr.append(0)
        return len(chunk)


arr = array("b", range(128))
arr.tofile(_ReenteringWriter(arr))
assert len(arr) == 129

# CPython 3.14 reconstructs foreign-width integers into a native typecode
# with the same width and signedness instead of narrowing through the
# originally pickled C type.
rebuilt = _array_reconstructor(
    array, "L", 6, b"\x01\x00\x00\x00\xff\xff\xff\xff"
)
assert rebuilt.typecode == "I"
assert rebuilt.tolist() == [1, 2**32 - 1]

# A short read that is not item-aligned is rejected by frombytes before the
# EOF check, and must not append the complete prefix.
partial = array("i")
with assert_raises(ValueError):
    partial.fromfile(BytesIO(b"\x01\x00\x00\x00X"), 2)
assert partial == array("i")
