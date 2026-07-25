"""`int.to_bytes(0, ...)` only accepts zero, including in signed mode."""


def raises_overflow(operation):
    try:
        operation()
    except OverflowError as exc:
        assert "too big to convert" in str(exc), str(exc)
    else:
        raise AssertionError("zero-length to_bytes accepted a nonzero value")


assert (0).to_bytes(0, "big") == b""
assert (0).to_bytes(0, "little") == b""
assert (0).to_bytes(0, "big", signed=True) == b""
assert (0).to_bytes(0, "little", signed=True) == b""

# -1 is the one negative whose two's complement emits no bytes at all, so it
# reaches the end of the conversion with an empty buffer instead of tripping
# the per-byte width check.
raises_overflow(lambda: (-1).to_bytes(0, "big", signed=True))
raises_overflow(lambda: (-1).to_bytes(0, "little", signed=True))
raises_overflow(lambda: (1).to_bytes(0, "big", signed=True))
raises_overflow(lambda: (-2).to_bytes(0, "big", signed=True))
raises_overflow(lambda: (-(2**70)).to_bytes(0, "big", signed=True))

# Unsigned negatives keep reporting the signedness error, not the width one.
try:
    (-1).to_bytes(0, "big")
except OverflowError as exc:
    assert "negative int to unsigned" in str(exc), str(exc)
else:
    raise AssertionError("unsigned to_bytes accepted a negative value")

# The smallest width that does fit still round-trips.
assert (-1).to_bytes(1, "big", signed=True) == b"\xff"
assert int.from_bytes((-1).to_bytes(1, "big", signed=True), "big", signed=True) == -1

print("OK")
