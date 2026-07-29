"""CPython 3.14 REDUCE operand validation."""

import pickle


cases = (
    (b"N)R.", "'NoneType' object is not callable"),
    (b"cbuiltins\nint\nNR.", "argument list must be a tuple"),
)

for payload, message in cases:
    try:
        pickle.loads(payload)
    except TypeError as exc:
        assert str(exc) == message
    else:
        raise AssertionError(f"accepted malformed pickle {payload!r}")

print("OK")
