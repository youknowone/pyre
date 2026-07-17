WRAPPER_DESCRIPTORS = {
    "__init__",
    "__iand__",
    "__ior__",
    "__isub__",
    "__ixor__",
}

METHOD_DESCRIPTORS = {
    "add",
    "clear",
    "difference_update",
    "discard",
    "intersection_update",
    "pop",
    "remove",
    "symmetric_difference_update",
    "update",
}

for name in WRAPPER_DESCRIPTORS:
    descriptor = set.__dict__[name]
    try:
        descriptor(frozenset())
    except TypeError as exc:
        assert str(exc) == (
            f"descriptor '{name}' requires a 'set' object but received a 'frozenset'"
        )
    else:
        raise AssertionError(f"set.{name} accepted a frozenset receiver")

    try:
        descriptor()
    except TypeError as exc:
        assert str(exc) == f"descriptor '{name}' of 'set' object needs an argument"
    else:
        raise AssertionError(f"set.{name} accepted a missing receiver")

for name in METHOD_DESCRIPTORS:
    descriptor = set.__dict__[name]
    try:
        descriptor(frozenset())
    except TypeError as exc:
        assert str(exc) == (
            f"descriptor '{name}' for 'set' objects doesn't apply to a 'frozenset' object"
        )
    else:
        raise AssertionError(f"set.{name} accepted a frozenset receiver")

    try:
        descriptor()
    except TypeError as exc:
        assert str(exc) == f"unbound method set.{name}() needs an argument"
    else:
        raise AssertionError(f"set.{name} accepted a missing receiver")


class SetSubclass(set):
    pass


value = SetSubclass((1, 2))
set.add(value, 3)
set.update(value, (4, 5))
set.difference_update(value, (1,))
set.intersection_update(value, (2, 3, 4))
set.symmetric_difference_update(value, (4, 6))
assert value == {2, 3, 6}
set.discard(value, 2)
set.remove(value, 3)
assert set.pop(value) == 6
set.clear(value)
assert value == set()

print("OK")
