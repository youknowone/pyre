iterator_type = type(iter(set()))

assert iterator_type.__name__ == "set_iterator"
assert set(iterator_type.__dict__) == {
    "__doc__",
    "__iter__",
    "__length_hint__",
    "__next__",
    "__reduce__",
}

WRAPPERS = {"__iter__", "__next__"}
METHODS = {"__length_hint__", "__reduce__"}

for name in WRAPPERS:
    descriptor = iterator_type.__dict__[name]
    try:
        descriptor(iter([]))
    except TypeError as exc:
        assert str(exc) == (
            f"descriptor '{name}' requires a 'set_iterator' object "
            "but received a 'list_iterator'"
        )
    else:
        raise AssertionError(f"set_iterator.{name} accepted a list iterator")

    try:
        descriptor()
    except TypeError as exc:
        assert str(exc) == (
            f"descriptor '{name}' of 'set_iterator' object needs an argument"
        )
    else:
        raise AssertionError(f"set_iterator.{name} accepted a missing receiver")

for name in METHODS:
    descriptor = iterator_type.__dict__[name]
    try:
        descriptor(iter([]))
    except TypeError as exc:
        assert str(exc) == (
            f"descriptor '{name}' for 'set_iterator' objects "
            "doesn't apply to a 'list_iterator' object"
        )
    else:
        raise AssertionError(f"set_iterator.{name} accepted a list iterator")

    try:
        descriptor()
    except TypeError as exc:
        assert str(exc) == (
            f"unbound method set_iterator.{name}() needs an argument"
        )
    else:
        raise AssertionError(f"set_iterator.{name} accepted a missing receiver")


values = {1, 2, 3}
iterator = iter(values)
assert iter(iterator) is iterator
assert iterator.__length_hint__() == 3

first = next(iterator)
reduced = iterator.__reduce__()
assert reduced[0] is iter
assert len(reduced) == 2
remaining = reduced[1][0]
assert isinstance(remaining, list)
assert set(remaining) == values - {first}
assert iterator.__length_hint__() == 2
assert set(iterator) == values - {first}
assert iterator.__length_hint__() == 0
assert iterator.__reduce__() == (iter, ([],))

values = {1, 2}
iterator = iter(values)
values.add(3)
for operation in (iterator.__next__, iterator.__reduce__):
    try:
        operation()
    except RuntimeError as exc:
        assert str(exc) == "Set changed size during iteration"
    else:
        raise AssertionError("set mutation did not invalidate iterator")
assert iterator.__length_hint__() == 0

print("OK")
