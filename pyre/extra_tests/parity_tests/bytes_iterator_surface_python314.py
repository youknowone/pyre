"""CPython 3.14 keeps producer-specific bytes iterator identities."""


def check(value, expected):
    iterator = iter(value)
    typ = type(iterator)
    assert typ.__name__ == expected
    assert typ.__module__ == "builtins"
    assert list(iterator) == list(value)


check(b"abc", "bytes_iterator")
check(bytearray(b"abc"), "bytearray_iterator")
print("OK")
