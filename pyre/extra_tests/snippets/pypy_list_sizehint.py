import __pypy__


def assert_shape(value, strategy, length, physical_size):
    assert __pypy__.strategy(value) == strategy
    assert len(value) == length
    assert __pypy__.list_get_physical_size(value) == physical_size


# pypy/objspace/std/listobject.py SizeListStrategy keeps no backing storage.
# Its first append selects the concrete strategy and consumes the exact hint.
for hint, value, strategy in [
    (13, 7, "IntegerListStrategy"),
    (5, 1.5, "FloatListStrategy"),
    (6, b"x", "BytesListStrategy"),
    (7, "x", "AsciiListStrategy"),
    (8, None, "ObjectListStrategy"),
]:
    items = __pypy__.newlist_hint(hint)
    assert_shape(items, "SizeListStrategy", 0, 0)
    items.append(value)
    assert_shape(items, strategy, 1, hint)


items = __pypy__.newlist_hint(13)
assert_shape(items.copy(), "SizeListStrategy", 0, 0)
assert_shape(items * 3, "SizeListStrategy", 0, 0)
items *= 0
assert_shape(items, "SizeListStrategy", 0, 0)
assert_shape(__pypy__.newlist_hint(13)[:], "EmptyListStrategy", 0, 0)


# EmptyListStrategy.clone retains the strategy object itself.  Since Size is
# the one per-list strategy instance, its mutable hint is shared by clones.
for clone in [lambda value: value.copy(), lambda value: value * 3, lambda value: [] + value]:
    items = __pypy__.newlist_hint(5)
    copied = clone(items)
    __pypy__.resizelist_hint(items, 9)
    copied.append(1)
    assert_shape(copied, "IntegerListStrategy", 1, 9)


items = []
__pypy__.resizelist_hint(items, 13)
assert_shape(items, "SizeListStrategy", 0, 0)
items.append(1)
assert_shape(items, "IntegerListStrategy", 1, 13)


items = [1, 2]
__pypy__.resizelist_hint(items, 10)
assert_shape(items, "IntegerListStrategy", 2, 17)
assert items == [1, 2]


# rpython/rtyper/lltypesystem/rlist.py _ll_list_resize_hint_really uses the
# same 0, 4, 8, 16, 25, ... capacity policy for every resizable strategy.
for value, strategy in [
    (1, "IntegerListStrategy"),
    (1.5, "FloatListStrategy"),
    (b"x", "BytesListStrategy"),
    ("x", "AsciiListStrategy"),
    (None, "ObjectListStrategy"),
]:
    items = []
    for length, physical_size in [(1, 4), (4, 4), (5, 8), (8, 8), (9, 16), (17, 25)]:
        while len(items) < length:
            items.append(value)
        assert_shape(items, strategy, length, physical_size)
