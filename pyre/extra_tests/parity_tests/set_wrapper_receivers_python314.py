WRAPPERS = {
    "__and__",
    "__eq__",
    "__ge__",
    "__gt__",
    "__iter__",
    "__le__",
    "__len__",
    "__lt__",
    "__ne__",
    "__or__",
    "__rand__",
    "__repr__",
    "__ror__",
    "__rsub__",
    "__rxor__",
    "__sub__",
    "__xor__",
}

for owner, wrong in ((set, frozenset()), (frozenset, set())):
    owner_name = owner.__name__
    wrong_name = type(wrong).__name__
    for name in WRAPPERS:
        descriptor = owner.__dict__[name]
        try:
            descriptor(wrong)
        except TypeError as exc:
            assert str(exc) == (
                f"descriptor '{name}' requires a '{owner_name}' object "
                f"but received a '{wrong_name}'"
            )
        else:
            raise AssertionError(f"{owner_name}.{name} accepted {wrong_name}")

        try:
            descriptor()
        except TypeError as exc:
            assert str(exc) == (
                f"descriptor '{name}' of '{owner_name}' object needs an argument"
            )
        else:
            raise AssertionError(f"{owner_name}.{name} accepted a missing receiver")

for args in ((), (set(),)):
    try:
        frozenset.__hash__(*args)
    except TypeError as exc:
        if args:
            assert str(exc) == (
                "descriptor '__hash__' requires a 'frozenset' object "
                "but received a 'set'"
            )
        else:
            assert str(exc) == (
                "descriptor '__hash__' of 'frozenset' object needs an argument"
            )
    else:
        raise AssertionError("frozenset.__hash__ accepted an invalid receiver")


class SetSubclass(set):
    def __iter__(self):
        return iter(("override",))

    def __repr__(self):
        return "override repr"


class FrozenSubclass(frozenset):
    def __iter__(self):
        return iter(("override",))

    def __repr__(self):
        return "override repr"


set_subclass = SetSubclass((1, 2))
frozen_subclass = FrozenSubclass((1, 2))
assert sorted(set.__iter__(set_subclass)) == [1, 2]
assert sorted(frozenset.__iter__(frozen_subclass)) == [1, 2]
assert set.__repr__(set_subclass) == "SetSubclass({1, 2})"
assert frozenset.__repr__(frozen_subclass) == "FrozenSubclass({1, 2})"
assert set.__len__(set_subclass) == 2
assert frozenset.__len__(frozen_subclass) == 2
assert isinstance(frozenset.__hash__(frozen_subclass), int)

print("OK")
