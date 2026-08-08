"""A builtin subclass's `__iter__` override wins over the storage iterator.

`objspace.py:507-527` takes the concrete fast path for `unpackiterable` and
`fixedview` only when `objspace.py:633-643 _uses_list_iter` / `_uses_tuple_iter`
/ `_uses_unicode_iter` says the receiver's `__iter__` is still the one the
builtin base defines; a tuple subtype keeps the fast path while it inherits,
and the list arm is an exact-type test.  `functional.py:268` guards the
`enumerate` shortcut with `type(w_iterable) is W_ListObject` for the same
reason.  Everything else goes through `space.iter`, which dispatches the
override.

Missing that guard is not only a wrong answer: the storage iterator reads the
subtype through the base layout.
"""

MARK = ["over"]


def override(base):
    class Sub(base):
        def __iter__(self):
            return iter(MARK)

    Sub.__name__ = f"{base.__name__.title()}Sub"
    return Sub


def collect(*a):
    return a


LIST = override(list)([1, 2])
TUPLE = override(tuple)((1, 2))
STR = override(str)("ab")
BYTES = override(bytes)(b"ab")
BYTEARRAY = override(bytearray)(b"ab")
SET = override(set)({1, 2})
FROZENSET = override(frozenset)({1, 2})
DICT = override(dict)(a=1)

for label, obj in [
    ("list", LIST),
    ("tuple", TUPLE),
    ("str", STR),
    ("bytes", BYTES),
    ("bytearray", BYTEARRAY),
    ("set", SET),
    ("frozenset", FROZENSET),
    ("dict", DICT),
]:
    assert list(iter(obj)) == MARK, (label, "iter")
    assert [x for x in obj] == MARK, (label, "for")
    assert [*obj] == MARK, (label, "unpack")
    assert collect(*obj) == ("over",), (label, "call unpack")
    assert list(obj) == MARK, (label, "list()")
    assert tuple(obj) == ("over",), (label, "tuple()")
    assert list(enumerate(obj)) == [(0, "over")], (label, "enumerate")
    assert list(enumerate(obj, 1)) == [(1, "over")], (label, "enumerate start")
    assert list(zip(obj, obj)) == [("over", "over")], (label, "zip")
    assert list(map(str, obj)) == MARK, (label, "map")
    assert sorted(obj) == MARK, (label, "sorted")
    assert min(obj) == "over", (label, "min")
    # `set()` and `frozenset()` copy a set argument's storage instead of
    # iterating it, so an override on a set subtype does not reach them.
    want_set = {1, 2} if label in ("set", "frozenset") else {"over"}
    assert set(obj) == want_set, (label, "set()")
    assert frozenset(obj) == want_set, (label, "frozenset()")

# `**` unpacking reads the mapping, not the iterator, so a dict subtype's
# `__iter__` does not reach it.
assert {**DICT} == {"a": 1}

# An inheriting subclass keeps the storage iterator and its own contents.
for base, made, want in [
    (list, [1, 2], [1, 2]),
    (tuple, (1, 2), [1, 2]),
    (str, "ab", ["a", "b"]),
    (dict, {"a": 1}, ["a"]),
]:
    plain = type("Plain", (base,), {})(made)
    assert [*plain] == want, (base, [*plain])
    assert collect(*plain) == tuple(want), (base, collect(*plain))
    assert list(enumerate(plain)) == list(enumerate(want)), base


# `__iter__ = None` marks the subclass non-iterable even though the lookup
# succeeds (`descroperation.py:339-341`).
class NoIter(list):
    __iter__ = None


for label, fn in [
    ("iter", lambda: iter(NoIter([1]))),
    ("for", lambda: [x for x in NoIter([1])]),
    ("unpack", lambda: [*NoIter([1])]),
    ("call unpack", lambda: collect(*NoIter([1]))),
    ("enumerate", lambda: list(enumerate(NoIter([1])))),
]:
    try:
        result = fn()
    except TypeError:
        continue
    raise AssertionError(f"{label} returned {result!r} instead of raising TypeError")

print("OK")
