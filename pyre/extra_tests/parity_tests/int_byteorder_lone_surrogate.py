# CPython-suite gap: test_long checks the byteorder argument only with the two
# accepted spellings and with plain ASCII junk, so nothing in the suite passes a
# string that has no UTF-8 encoding.
# parity-tests reason: the ValueError and its text are identical on CPython 3.14
# and PyPy 7.3.20, so all three runtimes are asserted against the same literal.

"""A byteorder that is not valid UTF-8 is rejected, not fatal.

``int.to_bytes`` and ``int.from_bytes`` compare their byteorder argument
against the literals ``"little"`` and ``"big"``.  Reading that argument as
``&str`` cannot represent a lone surrogate, so the projection has to be the
fallible one: an unrepresentable string simply matches neither literal and
takes the same arm as ``"middle"``.  Taking the infallible projection instead
ends the process before any Python-level handler runs, which no amount of
``except`` around the call can recover from.

The surrogate is built with ``chr`` rather than a literal escape so that the
string really is a lone surrogate at runtime and not a source-level artefact.
"""

LONE = chr(0xD800)


def rejects(fn):
    try:
        fn()
    except ValueError as exc:
        return str(exc)
    raise AssertionError("byteorder was accepted")


EXPECTED = "byteorder must be either 'little' or 'big'"

# The argument really is unrepresentable: it survives as a str and has no
# UTF-8 encoding, which is the whole precondition of this test.
assert isinstance(LONE, str) and len(LONE) == 1
try:
    LONE.encode("utf-8")
except UnicodeEncodeError:
    pass
else:
    raise AssertionError("expected a lone surrogate to be unencodable")

# Both accepted spellings still work, so the fallible read did not change the
# answer on the path that matches.
assert (1).to_bytes(2, "little") == b"\x01\x00"
assert (1).to_bytes(2, "big") == b"\x00\x01"
assert int.from_bytes(b"\x01\x02", "little") == 513
assert int.from_bytes(b"\x01\x02", "big") == 258

# An unrepresentable byteorder is rejected exactly like an unrecognised one.
for label, call in [
    ("to_bytes", lambda: (1).to_bytes(2, LONE)),
    ("from_bytes", lambda: int.from_bytes(b"\x01\x02", LONE)),
    ("to_bytes keyword", lambda: (1).to_bytes(2, byteorder=LONE)),
    ("from_bytes keyword", lambda: int.from_bytes(b"\x01\x02", byteorder=LONE)),
]:
    got = rejects(call)
    assert got == EXPECTED, f"{label}\n  expected {EXPECTED!r}\n  got      {got!r}"

# ... and an ordinary unrecognised spelling still reports the same thing, so
# the two arms have not drifted apart.
assert rejects(lambda: (1).to_bytes(2, "middle")) == EXPECTED

print("OK")
