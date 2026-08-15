from testutils import assert_raises, skip_if_unsupported


def test_dunion_ior0():
    a = {1: 2, 2: 3}
    b = {3: 4, 5: 6}
    a |= b

    assert a == {1: 2, 2: 3, 3: 4, 5: 6}, f"wrong value assigned {a=}"
    assert b == {3: 4, 5: 6}, f"right hand side modified, {b=}"


def test_dunion_or0():
    a = {1: 2, 2: 3}
    b = {3: 4, 5: 6}
    c = a | b

    assert a == {1: 2, 2: 3}, f"left hand side of non-assignment operator modified {a=}"
    assert b == {3: 4, 5: 6}, (
        f"right hand side of non-assignment operator modified, {b=}"
    )
    assert c == {1: 2, 2: 3, 3: 4, 5: 6}, f"unexpected result of dict union {c=}"


def test_dunion_or1():
    a = {1: 2, 2: 3}
    b = {3: 4, 5: 6}
    c = a.__or__(b)

    assert a == {1: 2, 2: 3}, f"left hand side of non-assignment operator modified {a=}"
    assert b == {3: 4, 5: 6}, (
        f"right hand side of non-assignment operator modified, {b=}"
    )
    assert c == {1: 2, 2: 3, 3: 4, 5: 6}, f"unexpected result of dict union {c=}"


def test_dunion_ror0():
    a = {1: 2, 2: 3}
    b = {3: 4, 5: 6}
    c = b.__ror__(a)

    assert a == {1: 2, 2: 3}, f"left hand side of non-assignment operator modified {a=}"
    assert b == {3: 4, 5: 6}, (
        f"right hand side of non-assignment operator modified, {b=}"
    )
    assert c == {1: 2, 2: 3, 3: 4, 5: 6}, f"unexpected result of dict union {c=}"


def test_dunion_other_types():
    def perf_test_or(other_obj):
        d = {1: 2}
        return d.__or__(other_obj) is NotImplemented

    def perf_test_ror(other_obj):
        d = {1: 2}
        return d.__ror__(other_obj) is NotImplemented

    test_fct = {"__or__": perf_test_or, "__ror__": perf_test_ror}
    others = ["FooBar", 42, [36], set([19]), ["aa"], None]
    for tfn, tf in test_fct.items():
        for other in others:
            assert tf(other), f"Failed: dict {tfn}, accepted {other}"

    # __ior__() has different behavior and needs to be tested separately
    d = {1: 2}
    assert_raises(
        ValueError,
        lambda: d.__ior__("FooBar"),
        _msg="dictionary update sequence element #0 has length 1; 2 is required",
    )
    assert_raises(TypeError, lambda: d.__ior__(42), _msg="'int' object is not iterable")
    assert_raises(
        TypeError,
        lambda: d.__ior__([36]),
        _msg="cannot convert dictionary update sequence element #0 to a sequence",
    )
    assert_raises(
        TypeError,
        lambda: d.__ior__(set([36])),
        _msg="cannot convert dictionary update sequence element #0 to a sequence",
    )
    res = d.__ior__(["aa"])
    assert res == {1: 2, "a": "a"}, f"unexpected result of dict union {res=}"
    assert_raises(
        TypeError,
        lambda: d.__ior__(None),
        _msg="TypeError: 'NoneType' object is not iterable",
    )


skip_if_unsupported(3, 9, test_dunion_ior0)
skip_if_unsupported(3, 9, test_dunion_or0)
skip_if_unsupported(3, 9, test_dunion_or1)
skip_if_unsupported(3, 9, test_dunion_ror0)
skip_if_unsupported(3, 9, test_dunion_other_types)


def test_dunion_preserves_stored_hashes():
    # `d | other` copies the left operand and updates the copy from the right,
    # and a dict-to-dict update reuses each key's stored hash.  Neither half
    # re-invokes `__hash__`, which is observable for a key that records it.
    calls = []

    class Key:
        def __init__(self, i):
            self.i = i

        def __hash__(self):
            calls.append(self.i)
            return self.i

        def __eq__(self, other):
            return isinstance(other, Key) and self.i == other.i

    left = {Key(i): i for i in range(5)}
    right = {Key(50 + i): i for i in range(3)}

    calls.clear()
    merged = left | {}
    assert calls == [], f"left operand was re-hashed: {calls=}"
    assert len(merged) == 5, f"unexpected merge result {merged=}"

    calls.clear()
    merged = left | right
    assert calls == [], f"merge re-hashed an operand: {calls=}"
    assert len(merged) == 8, f"unexpected merge result {merged=}"

    # The same reuse applies to `update` into a destination that is not empty,
    # which is the path the merge above takes for its right operand.
    destination = {Key(99): 0}
    calls.clear()
    destination.update(left)
    assert calls == [], f"update re-hashed the source: {calls=}"
    assert len(destination) == 6, f"unexpected update result {destination=}"

    # Typed-strategy sources carry no stored hash and take the plain walk.
    assert {1: "a"} | {2: "b"} == {1: "a", 2: "b"}
    assert {"x": 1} | {"y": 2} == {"x": 1, "y": 2}


skip_if_unsupported(3, 9, test_dunion_preserves_stored_hashes)
