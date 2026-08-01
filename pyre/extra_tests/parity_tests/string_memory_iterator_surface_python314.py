"""CPython 3.14 gives text and memoryview iterators concrete identities."""


def check(value, expected):
    iterator = iter(value)
    typ = type(iterator)
    assert typ.__name__ == expected
    assert typ.__module__ == "builtins"
    assert list(iterator) == list(value)


check("abc", "str_ascii_iterator")
check("é", "str_iterator")
check(memoryview(b"abc"), "memory_iterator")
assert sorted(type(iter(memoryview(b""))).__dict__) == ["__doc__", "__iter__", "__next__"]
print("OK")
