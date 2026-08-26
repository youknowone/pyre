# pyre-check: gate=1
# The `start` / `end` arguments to a `Unicode*Error` are `Py_ssize_t`, so the
# constructor converts them through the index protocol and stores the number.
# `args` is untouched by that conversion and keeps what the caller passed,
# which is the same split `object` already has: the slot holds the coerced
# value while `args` holds the original.

class MyInt(int):
    pass

class Indexable:
    def __index__(self):
        return 3

def errors(kind, start, end=1):
    if kind == "decode":
        return UnicodeDecodeError("utf-8", b"xy", start, end, "r")
    if kind == "encode":
        return UnicodeEncodeError("utf-8", "xy", start, end, "r")
    return UnicodeTranslateError("xy", start, end, "r")

for kind in ("decode", "encode", "translate"):
    # The stored position is a plain int whatever shape it arrived in.
    for value, expected in ((1, 1), (True, 1), (MyInt(0), 0), (Indexable(), 3), (-5, -5)):
        exc = errors(kind, value)
        assert exc.start == expected, (kind, value, exc.start)
        assert type(exc.start) is int, (kind, value, type(exc.start))

    # Anything the index protocol cannot answer is refused, and names itself.
    for value, name in (("x", "str"), (None, "NoneType"), (1.0, "float")):
        try:
            errors(kind, value)
        except TypeError as error:
            assert str(error) == f"'{name}' object cannot be interpreted as an integer", (
                kind, value, str(error),
            )
        else:
            raise AssertionError(f"{kind} accepted a {name} position")

    # A value past the platform word is an OverflowError, not a stored bignum.
    try:
        errors(kind, 10 ** 30)
    except OverflowError as error:
        assert str(error) == "Python int too large to convert to C ssize_t", str(error)
    else:
        raise AssertionError(f"{kind} accepted an oversize position")

# `args` keeps the objects the caller handed over, unconverted.
exc = UnicodeDecodeError("utf-8", bytearray(b"xy"), True, MyInt(1), "r")
assert exc.args == ("utf-8", bytearray(b"xy"), True, 1, "r"), exc.args
assert [type(a).__name__ for a in exc.args] == [
    "str", "bytearray", "bool", "MyInt", "str",
], [type(a).__name__ for a in exc.args]
# ... while the slots hold the converted values.
assert type(exc.start) is int and type(exc.end) is int
assert exc.object == b"xy" and type(exc.object) is bytes

# An argument of the wrong type names that type rather than a placeholder.
for call, message in (
    (lambda: UnicodeDecodeError(1, b"xy", 0, 1, "r"), "argument 1 must be str, not int"),
    (lambda: UnicodeDecodeError("utf-8", 1, 0, 1, "r"),
     "a bytes-like object is required, not 'int'"),
    (lambda: UnicodeDecodeError("utf-8", b"xy", 0, 1, 1), "argument 5 must be str, not int"),
    (lambda: UnicodeEncodeError(1, "xy", 0, 1, "r"), "argument 1 must be str, not int"),
    (lambda: UnicodeEncodeError("utf-8", 1, 0, 1, "r"), "argument 2 must be str, not int"),
    (lambda: UnicodeEncodeError("utf-8", "xy", 0, 1, 1), "argument 5 must be str, not int"),
    (lambda: UnicodeTranslateError(1, 0, 1, "r"), "argument 1 must be str, not int"),
    (lambda: UnicodeTranslateError("xy", 0, 1, 1), "argument 4 must be str, not int"),
):
    try:
        call()
    except TypeError as error:
        assert str(error) == message, (str(error), message)
    else:
        raise AssertionError(f"expected {message!r}")
