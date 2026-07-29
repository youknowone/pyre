"""CPython 3.14 BUILD state operand validation."""

import pickle


class Value:
    pass


prefix = b"c__main__\nValue\n)\x81"
cases = (
    (
        prefix + b"}}NNs\x86b.",
        TypeError,
        "attribute name must be string, not 'NoneType'",
    ),
    (
        prefix + b"}\x88\x86b.",
        pickle.UnpicklingError,
        "slot state is not a dictionary",
    ),
)

for payload, error_type, message in cases:
    try:
        pickle.loads(payload)
    except error_type as exc:
        assert str(exc) == message
    else:
        raise AssertionError(f"accepted malformed pickle {payload!r}")

print("OK")
