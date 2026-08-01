def unpack_three(value):
    a, b, c = value
    return a, b, c


def error(value):
    try:
        unpack_three(value)
    except ValueError as exc:
        return str(exc)
    raise AssertionError("unpack unexpectedly succeeded")


assert error({"a": 1, "b": 2}) == (
    "not enough values to unpack (expected 3, got 2)"
)
assert error({"a": 1, "b": 2, "c": 3, "d": 4}) == (
    "too many values to unpack (expected 3, got 4)"
)
assert unpack_three({"a": 1, "b": 2, "c": 3}) == ("a", "b", "c")


class DictSubclass(dict):
    def __len__(self):
        raise AssertionError("unpacking must not call a dict subclass's __len__")


assert error(DictSubclass(a=1, b=2, c=3, d=4)) == (
    "too many values to unpack (expected 3)"
)

print("OK")
