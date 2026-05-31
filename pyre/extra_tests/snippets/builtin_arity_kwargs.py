# Arity / kwargs robustness for builtin methods (raise TypeError, no panic;
# parse keyword arguments instead of treating the kwargs dict as a value).

# dict.fromkeys requires the iterable argument.
try:
    dict.fromkeys()
    raise AssertionError("expected TypeError")
except TypeError as e:
    assert "at least 1 argument" in str(e), str(e)
assert dict.fromkeys([1, 2]) == {1: None, 2: None}
assert dict.fromkeys([1], 0) == {1: 0}

# str/bytes/bytearray.removeprefix/removesuffix take exactly one argument.
for bad in (lambda: "abc".removeprefix(),
            lambda: "abc".removeprefix("a", "b"),
            lambda: b"abc".removeprefix(),
            lambda: bytearray(b"abc").removesuffix()):
    try:
        bad()
        raise AssertionError("expected TypeError")
    except TypeError as e:
        assert "takes exactly one argument" in str(e), str(e)
assert "abc".removeprefix("a") == "bc"
assert "abc".removesuffix("c") == "ab"
assert b"abc".removeprefix(b"a") == b"bc"
assert bytearray(b"abc").removesuffix(b"c") == bytearray(b"ab")

# splitlines: keepends is positional-or-keyword, not the kwargs dict.
assert "a\nb".splitlines(keepends=False) == ["a", "b"]
assert "a\nb".splitlines(keepends=True) == ["a\n", "b"]
assert b"a\nb".splitlines(keepends=False) == [b"a", b"b"]
assert b"a\nb".splitlines(keepends=True) == [b"a\n", b"b"]
assert "a\nb".splitlines(True) == ["a\n", "b"]

# A default __init_subclass__ rejects leftover class-definition keywords.
class Base:
    pass
try:
    class C(Base, flag=1):
        pass
    raise AssertionError("expected TypeError")
except TypeError as e:
    assert "__init_subclass__" in str(e), str(e)

# A user __init_subclass__ still receives the keywords.
seen = {}
class Base2:
    def __init_subclass__(cls, /, **kw):
        seen.update(kw)
class C2(Base2, flag=7):
    pass
assert seen == {"flag": 7}, seen

print("builtin_arity_kwargs ok")
