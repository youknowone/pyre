# pyre-check: gate=1
# `charmap_encode` routes characters the table leaves undefined through the
# error handler registered under `errors`, rather than deciding for itself.
# A run of undefined characters is reported as one span, and a `str`
# replacement is mapped through the table in turn -- which is why using
# `"replace"` with a charmap requires the table to have an entry for `?`.

import codecs

TABLE = {ord(c): bytes(2 * c.upper(), "ascii") for c in "abcdefgh"}

assert codecs.charmap_encode("abc", "strict", TABLE)[0] == b"AABBCC"

# The replacement goes through the table, so it fails while `?` is unmapped
# and succeeds once the table can encode it.
try:
    codecs.charmap_encode("abcDEF", "replace", TABLE)
except UnicodeEncodeError:
    pass
else:
    raise AssertionError("an unmapped '?' should have refused the replacement")

TABLE[ord("?")] = b"XYZ"
assert codecs.charmap_encode("abcDEF", "replace", TABLE)[0] == b"AABBCCXYZXYZXYZ"

# A table value of the wrong type is a TypeError, not an encode error.
TABLE[ord("?")] = "XYZ"
try:
    codecs.charmap_encode("abcDEF", "replace", TABLE)
except TypeError:
    pass
else:
    raise AssertionError("a str table value should have been refused")
del TABLE[ord("?")]

# The raised error carries the codec name, the whole input, and the span.
try:
    codecs.charmap_encode("\xff", "strict", {})
except UnicodeEncodeError as exc:
    assert exc.encoding == "charmap", exc.encoding
    assert exc.object == "\xff", exc.object
    assert (exc.start, exc.end) == (0, 1), (exc.start, exc.end)
    assert exc.reason == "character maps to <undefined>", exc.reason
    assert str(exc), "the error should render a message"
else:
    raise AssertionError("an empty table should have refused '\\xff'")

# Consecutive undefined characters collapse into a single span.
seen = []

def spy(exc):
    seen.append((exc.start, exc.end, exc.object, exc.encoding, exc.reason))
    return ("", exc.end)

codecs.register_error("pyre.charmap.spy", spy)
assert codecs.charmap_encode("a\xff\xfe\xfdb", "pyre.charmap.spy",
                             {ord("a"): b"A", ord("b"): b"B"}) == (b"AB", 5)
assert seen == [(1, 4, "a\xff\xfe\xfdb", "charmap",
                 "character maps to <undefined>")], seen

try:
    codecs.charmap_encode("\xff\xfe\xfd", "strict", {})
except UnicodeEncodeError as exc:
    assert (exc.start, exc.end) == (0, 3), (exc.start, exc.end)

# A handler returning a result of the wrong shape is a TypeError.
codecs.register_error("pyre.charmap.bad", lambda exc: 42)
try:
    "\u3042".encode("iso-8859-15", "pyre.charmap.bad")
except TypeError:
    pass
else:
    raise AssertionError("a non-tuple handler result should have been refused")

# A handler whose replacement is itself unencodable reports the span that was
# originally undefined, not the replacement's own position.
class Rewinding:
    def __init__(self):
        self.count = 0
    def __call__(self, exc):
        if self.count > 0:
            self.count -= 1
            return ("\udcff", 0)
        return ("\udcff", exc.end)

rewinding = Rewinding()
codecs.register_error("pyre.charmap.rewind", rewinding)
rewinding.count = 5
try:
    "abcd\udc80".encode("iso-8859-15", "pyre.charmap.rewind")
except UnicodeEncodeError as exc:
    assert (exc.start, exc.end) == (4, 5), (exc.start, exc.end)
    assert exc.object == "abcd\udc80", exc.object
else:
    raise AssertionError("an unencodable replacement should have raised")

# `ignore` drops the run and still reports the whole input as consumed.
assert codecs.charmap_encode("\xff", "ignore", {0xFF: None}) == (b"", 1)

# A mapping raising something of its own is not read as "undefined".
class Raising(dict):
    def __getitem__(self, key):
        raise ValueError("boom")

try:
    codecs.charmap_encode("\xff", "strict", Raising())
except ValueError as error:
    assert str(error) == "boom", str(error)
else:
    raise AssertionError("the mapping's own error should have propagated")

# The table's value is read with `bytes`: an integer, a `bytes` and `None` are
# what a mapping may return, and a `bytearray` is none of the three.
try:
    codecs.charmap_encode("a", "strict", {97: bytearray(b"x")})
except TypeError as error:
    assert "integer, bytes or None" in str(error), str(error)
else:
    raise AssertionError("a bytearray table value should be a TypeError")

assert codecs.charmap_encode("a", "strict", {97: b"x"}) == (b"x", 1)
assert codecs.charmap_encode("a", "strict", {97: 120}) == (b"x", 1)

# The encode leg reads a handler position the way the decode leg does, and an
# unrepresentable one leaves through the same conversion error.
def overflowing(exc):
    return ("?", 10**100)

codecs.register_error("pyre.charmap.encode.overflow", overflowing)
try:
    codecs.charmap_encode("ሴ", "pyre.charmap.encode.overflow", {})
except OverflowError:
    pass
else:
    raise AssertionError("an unrepresentable position should overflow")
