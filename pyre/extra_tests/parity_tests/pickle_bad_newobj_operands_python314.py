"""CPython 3.14 NEWOBJ and NEWOBJ_EX operand validation."""

import pickle


cases = (
    (
        b"cbuiltins\nlen\n)\x81.",
        "NEWOBJ class argument must be a type, not builtin_function_or_method",
    ),
    (
        b"cbuiltins\nint\nN\x81.",
        "NEWOBJ args argument must be a tuple, not NoneType",
    ),
    (
        b"cbuiltins\nlen\n)}\x92.",
        "NEWOBJ_EX class argument must be a type, not builtin_function_or_method",
    ),
    (
        b"cbuiltins\nint\nN}\x92.",
        "NEWOBJ_EX args argument must be a tuple, not NoneType",
    ),
    (
        b"cbuiltins\nint\n)N\x92.",
        "NEWOBJ_EX kwargs argument must be a dict, not NoneType",
    ),
)

for payload, message in cases:
    try:
        pickle.loads(payload)
    except pickle.UnpicklingError as exc:
        assert str(exc) == message
    else:
        raise AssertionError(f"accepted malformed pickle {payload!r}")

print("OK")
