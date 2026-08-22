# CPython-suite gap: no test feeds marshal/pickle a three-byte sequence that
# holds no code point, and none indexes _json past the subject's length.
# parity-tests reason: these reach pyre's own WTF-8 representation, where the
# rejected buffer used to become a str whose length disagrees with its bytes.

"""Bytes from outside the runtime are checked before they become a str."""

import json
import marshal
import pickle
import _json


def raises(exc, fn):
    try:
        fn()
    except exc as caught:
        return str(caught)
    raise AssertionError(f"{exc.__name__} not raised")


# `ED C0 80` and `ED A0 41` pass a surrogate check that bounds only the first
# byte, and neither encodes a code point.
for payload in (b"\xed\xc0\x80", b"\xed\xa0\x41"):
    reason = "'utf-8' codec can't decode byte 0xed in position 0: invalid continuation byte"
    assert raises(UnicodeDecodeError, lambda: marshal.loads(b"u\x03\x00\x00\x00" + payload)) == reason
    assert raises(UnicodeDecodeError, lambda: pickle.loads(b"\x80\x04\x8c\x03" + payload + b".")) == reason

# The same encoding of a real lone surrogate stays a one-character string.
assert marshal.loads(b"u\x03\x00\x00\x00\xed\xa0\x80") == "\ud800"
assert pickle.loads(b"\x80\x04\x8c\x03\xed\xa0\x80.") == "\ud800"
# A pair stays two code points, which is what `surrogatepass` decodes.
assert marshal.loads(b"u\x06\x00\x00\x00\xed\xa0\x80\xed\xb0\x80") == "\ud800\udc00"

# `end` is a code point index into the subject, and both entry points bound it
# before resolving it to a byte offset.
subject = "中" * 100
assert raises(StopIteration, lambda: json.JSONDecoder().scan_once(subject, 200)) == "200"
assert raises(ValueError, lambda: _json.scanstring(subject, 200)) == "end is out of bounds"
assert _json.scanstring('"ab"', 1) == ("ab", 4)

# A lone surrogate is a length-1 separator with no UTF-8 spelling.
assert raises(ValueError, lambda: b"ab".hex(chr(0xDC80))) == "sep must be ASCII."
assert b"ab".hex("-") == "61-62"

# `fromhex` rejects one as an ordinary non-hex character.  Everything before
# the first rejected character is ASCII, so its byte offset is its index.
for subject, position in ((chr(0xDC80), 0), ("41" + chr(0xDC80) + "42", 2), ("41\u4e2d", 2)):
    reason = f"non-hexadecimal number found in fromhex() arg at position {position}"
    assert raises(ValueError, lambda: bytes.fromhex(subject)) == reason
    assert raises(ValueError, lambda: bytearray.fromhex(subject)) == reason
assert bytes.fromhex("41 42") == b"AB"

print("OK")
