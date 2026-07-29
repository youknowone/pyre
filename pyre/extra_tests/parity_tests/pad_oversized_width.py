import sys


# Each of these derives its buffer size from a `width` that arrives straight
# from Python, so an infallible reservation would abort the process instead of
# unwinding.  PyPy answers all of them with MemoryError; CPython agrees except
# where the result would be a `bytes`, which it rejects earlier with
# OverflowError("byte string is too large").
huge = sys.maxsize

cases = [
    ("str.ljust", MemoryError, lambda: "a".ljust(huge)),
    ("str.rjust", MemoryError, lambda: "a".rjust(huge)),
    ("str.center", MemoryError, lambda: "a".center(huge)),
    ("str.zfill", MemoryError, lambda: "a".zfill(huge)),
    ("bytes.ljust", (MemoryError, OverflowError), lambda: b"a".ljust(huge)),
    ("bytes.rjust", (MemoryError, OverflowError), lambda: b"a".rjust(huge)),
    ("bytes.center", (MemoryError, OverflowError), lambda: b"a".center(huge)),
    ("bytes.zfill", (MemoryError, OverflowError), lambda: b"a".zfill(huge)),
    ("bytes(n)", (MemoryError, OverflowError), lambda: bytes(huge)),
    ("bytearray.ljust", MemoryError, lambda: bytearray(b"a").ljust(huge)),
    ("bytearray.rjust", MemoryError, lambda: bytearray(b"a").rjust(huge)),
    ("bytearray.center", MemoryError, lambda: bytearray(b"a").center(huge)),
    ("bytearray.zfill", MemoryError, lambda: bytearray(b"a").zfill(huge)),
    ("bytearray(n)", MemoryError, lambda: bytearray(huge)),
    ("%s width", MemoryError, lambda: "%*s" % (huge, "a")),
    ("%d width", MemoryError, lambda: "%*d" % (huge, 1)),
    ("%f width", MemoryError, lambda: "%*f" % (huge, 1.0)),
    ("bytes %s width", MemoryError, lambda: b"%*s" % (huge, b"a")),
]

for label, expected, call in cases:
    try:
        call()
    except expected:
        pass
    else:
        raise AssertionError(f"{label} accepted an unsatisfiable width")

# A width that wraps when scaled to a worst-case per-code-point byte count
# must still be rejected rather than silently under-reserved.
for width in (2**62, 2**62 + 1, 2**61):
    try:
        "a".rjust(width)
    except MemoryError:
        pass
    else:
        raise AssertionError(f"rjust({width}) accepted an unsatisfiable width")

# Ordinary widths keep working.
assert "a".ljust(4, "-") == "a---"
assert "a".rjust(4, "-") == "---a"
assert "a".center(5, "-") == "--a--"
assert "-42".zfill(6) == "-00042"
assert b"a".ljust(4, b"-") == b"a---"
assert b"a".rjust(4, b"-") == b"---a"
assert b"a".center(5, b"-") == b"--a--"
assert b"-42".zfill(6) == b"-00042"
assert bytes(3) == b"\x00\x00\x00"
assert bytearray(3) == bytearray(b"\x00\x00\x00")
assert "%*s" % (4, "a") == "   a"
assert "%*d" % (4, 1) == "   1"
assert b"%*s" % (4, b"a") == b"   a"

print("OK")
