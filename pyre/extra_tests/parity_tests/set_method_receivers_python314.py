METHODS = {
    "__contains__",
    "__reduce__",
    "__sizeof__",
    "copy",
    "difference",
    "intersection",
    "isdisjoint",
    "issubset",
    "issuperset",
    "symmetric_difference",
    "union",
}

for owner, wrong in ((set, frozenset()), (frozenset, set())):
    owner_name = owner.__name__
    wrong_name = type(wrong).__name__
    for name in METHODS:
        descriptor = owner.__dict__[name]
        try:
            descriptor(wrong)
        except TypeError as exc:
            assert str(exc) == (
                f"descriptor '{name}' for '{owner_name}' objects "
                f"doesn't apply to a '{wrong_name}' object"
            )
        else:
            raise AssertionError(f"{owner_name}.{name} accepted {wrong_name}")

        try:
            descriptor()
        except TypeError as exc:
            assert str(exc) == (
                f"unbound method {owner_name}.{name}() needs an argument"
            )
        else:
            raise AssertionError(f"{owner_name}.{name} accepted a missing receiver")


class SetSubclass(set):
    pass


class FrozenSubclass(frozenset):
    pass


set_subclass = SetSubclass((1, 2))
frozen_subclass = FrozenSubclass((1, 2))

assert set.__contains__(set_subclass, 1)
assert frozenset.__contains__(frozen_subclass, 1)
assert set.union(set_subclass, (3,)) == {1, 2, 3}
assert frozenset.union(frozen_subclass, (3,)) == frozenset((1, 2, 3))
assert set.intersection(set_subclass, (2, 3)) == {2}
assert frozenset.intersection(frozen_subclass, (2, 3)) == frozenset((2,))
assert set.difference(set_subclass, (2,)) == {1}
assert frozenset.difference(frozen_subclass, (2,)) == frozenset((1,))
assert set.symmetric_difference(set_subclass, (2, 3)) == {1, 3}
assert frozenset.symmetric_difference(frozen_subclass, (2, 3)) == frozenset((1, 3))
assert set.issubset(set_subclass, (1, 2, 3))
assert frozenset.issubset(frozen_subclass, (1, 2, 3))
assert set.issuperset(set_subclass, (1,))
assert frozenset.issuperset(frozen_subclass, (1,))
assert set.isdisjoint(set_subclass, (3, 4))
assert frozenset.isdisjoint(frozen_subclass, (3, 4))
assert type(set.copy(set_subclass)) is set
assert type(frozenset.copy(frozen_subclass)) is frozenset
assert isinstance(set.__sizeof__(set_subclass), int)
assert isinstance(frozenset.__sizeof__(frozen_subclass), int)
assert isinstance(set.__reduce__(set_subclass), tuple)
assert isinstance(frozenset.__reduce__(frozen_subclass), tuple)

print("OK")
