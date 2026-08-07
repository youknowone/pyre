"""`str.ljust` / `str.rjust` / `str.center` convert their fill operand differently.

`descr_center` (`unicodeobject.py:1099-1101`) reads the operand with
`space.utf8_w`, which takes a `str` and nothing else.  `descr_ljust` and
`descr_rjust` (`unicodeobject.py:1352,1371`) go through
`convert_arg_to_w_unicode` (`unicodeobject.py:175-184`) instead: it declines
`bytes` with its own wording and hands anything else to `decode_object`
(`unicodeobject.py:1727-1739`), which reads the operand as a buffer and decodes
it, so a `bytearray` or `memoryview` becomes a fill character rather than a
refusal.  The length check that follows applies to the decoded result.

The reference refuses every non-`str` operand, so the two buffer rows are
asserted only off it.
"""

import sys


def raises_type_error(label, fn):
    try:
        result = fn()
    except TypeError:
        return
    raise AssertionError(f"{label} returned {result!r} instead of raising TypeError")


assert "x".ljust(5, "-") == "x----"
assert "x".rjust(5, "-") == "----x"
assert "x".center(5, "-") == "--x--"
assert "x".ljust(5) == "x    "
assert "abc".ljust(2, "-") == "abc"

# A multi-character fill is refused by length, whatever its type.
raises_type_error("'x'.ljust(5, '--')", lambda: "x".ljust(5, "--"))
raises_type_error("'x'.rjust(5, '')", lambda: "x".rjust(5, ""))
raises_type_error("'x'.center(5, '--')", lambda: "x".center(5, "--"))

# `bytes` is declined by name before any decode is attempted.
raises_type_error("'x'.ljust(5, b'-')", lambda: "x".ljust(5, b"-"))
raises_type_error("'x'.rjust(5, b'-')", lambda: "x".rjust(5, b"-"))

# A non-buffer operand has nothing to decode.
raises_type_error("'x'.ljust(5, None)", lambda: "x".ljust(5, None))
raises_type_error("'x'.rjust(5, 1)", lambda: "x".rjust(5, 1))

# `center` reads its operand strictly, so a buffer is refused there.
raises_type_error("'x'.center(5, bytearray(b'-'))", lambda: "x".center(5, bytearray(b"-")))
raises_type_error("'x'.center(5, memoryview(b'-'))", lambda: "x".center(5, memoryview(b"-")))

if sys.implementation.name == "cpython":
    raises_type_error("'x'.ljust(5, bytearray(b'-'))", lambda: "x".ljust(5, bytearray(b"-")))
    raises_type_error("'x'.rjust(5, memoryview(b'-'))", lambda: "x".rjust(5, memoryview(b"-")))
else:
    assert "x".ljust(5, bytearray(b"-")) == "x----"
    assert "x".rjust(5, memoryview(b"-")) == "----x"
    assert "x".ljust(5, bytearray("é", "utf-8")) == "xéééé"
    # The decoded operand still has to be a single character.
    raises_type_error(
        "'x'.ljust(5, bytearray(b'--'))", lambda: "x".ljust(5, bytearray(b"--"))
    )
    # The decode is strict, so a buffer that is not valid UTF-8 raises from it.
    try:
        "x".ljust(5, bytearray(b"\xff"))
    except UnicodeDecodeError:
        pass
    else:
        raise AssertionError("'x'.ljust(5, bytearray(b'\\xff')) did not raise")

print("OK")
