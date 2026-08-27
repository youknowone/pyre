# pyre-check: gate=1
# The decode leg of `charmap_encode_error_handler`: a byte the table leaves
# undefined goes to the registered handler, and the table itself may be an
# object with a `__getitem__` of its own.  Both run Python from inside the
# decode loop, which is also what makes this a pressure test for the input and
# the table staying reachable across those calls.

import codecs

TABLE = {i: chr(i) for i in range(256) if i != 0xDD}

assert codecs.charmap_decode(b"abc", "strict", TABLE) == ("abc", 3)

# An undefined byte reaches the handler with the codec name and the span.
seen = []

def spy(exc):
    seen.append((exc.encoding, exc.object, exc.start, exc.end, exc.reason))
    return ("<?>", exc.end)

codecs.register_error("pyre.charmap.decode.spy", spy)
assert codecs.charmap_decode(b"a\xddb", "pyre.charmap.decode.spy", TABLE) == ("a<?>b", 3)
assert seen == [("charmap", b"a\xddb", 1, 2, "character maps to <undefined>")], seen

# The builtin names still answer.
assert codecs.charmap_decode(b"a\xddb", "ignore", TABLE) == ("ab", 3)
assert codecs.charmap_decode(b"a\xddb", "replace", TABLE) == ("a�b", 3)
try:
    codecs.charmap_decode(b"a\xddb", "strict", TABLE)
except UnicodeDecodeError as exc:
    assert exc.encoding == "charmap", exc.encoding
    assert (exc.start, exc.end) == (1, 2), (exc.start, exc.end)
else:
    raise AssertionError("strict should have refused the undefined byte")

# A table raising something of its own is not read as "undefined".
class Raising(dict):
    def __getitem__(self, key):
        raise ValueError("boom")

try:
    codecs.charmap_decode(b"a", "strict", Raising())
except ValueError as error:
    assert str(error) == "boom", str(error)
else:
    raise AssertionError("the table's own error should have propagated")

# Pressure: many handler calls, each allocating, over a table and an input that
# both have to stay reachable for the whole decode.
def churn(exc):
    _ = [bytearray(64) for _ in range(400)]
    return ("y" * 300, exc.end)

codecs.register_error("pyre.charmap.decode.churn", churn)
data = (b"a" * 20 + b"\xdd") * 40
for _ in range(30):
    decoded, consumed = codecs.charmap_decode(data, "pyre.charmap.decode.churn", TABLE)
    assert consumed == len(data), (consumed, len(data))
    assert decoded.count("y") == 300 * 40, decoded.count("y")

# The same through the stdlib charmap codec, whose table is a real dict.
for _ in range(30):
    assert data.decode("iso-8859-15", "pyre.charmap.decode.churn") is not None

# A handler position too large for the machine integer is refused by the
# conversion that read it, so what comes out is the `OverflowError` the tuple's
# integer conversion raises rather than a bounds error over a substituted
# value.  The negative fold runs only on the branch where that conversion
# succeeded: fold a failure's -1 instead and a one-byte input reaches 0, which
# hands the same span back to the handler without end.
def overflowing(exc):
    return ("?", 10**100)

codecs.register_error("pyre.charmap.decode.overflow", overflowing)
try:
    codecs.charmap_decode(b"\xdd", "pyre.charmap.decode.overflow", TABLE)
except OverflowError:
    pass
else:
    raise AssertionError("an unrepresentable position should overflow")

# A position the conversion does read is still folded against the length.
def negative(exc):
    return ("?", -5)

codecs.register_error("pyre.charmap.decode.negative", negative)
try:
    codecs.charmap_decode(b"\xdd", "pyre.charmap.decode.negative", TABLE)
except IndexError as error:
    assert "position -4" in str(error), str(error)
else:
    raise AssertionError("a position before the input should be out of bounds")
