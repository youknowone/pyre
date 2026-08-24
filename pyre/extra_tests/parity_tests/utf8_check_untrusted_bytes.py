# pyre-check: pypy-diverges: pins _json.scanstring indexed past the subject's length; pypy3 has no _json module
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


def loads_both(payload):
    """Feed one `TYPE_UNICODE` / `SHORT_BINUNICODE` payload to both readers."""
    size = len(payload)
    yield lambda: marshal.loads(b"u" + size.to_bytes(4, "little") + payload)
    yield lambda: pickle.loads(b"\x80\x04\x8c" + bytes([size]) + payload + b".")


# `ED C0 80` and `ED A0 41` pass a surrogate check that bounds only the first
# byte, and neither encodes a code point.  Both readers decode with
# `surrogatepass`, so the position they report is the one that decode stops at
# -- not the first byte a strict scan trips over, which is the surrogate they
# accept.
for payload, reason in (
    (b"\xed\xc0\x80", "byte 0xed in position 0: invalid continuation byte"),
    (b"\xed\xa0\x41", "byte 0xed in position 0: invalid continuation byte"),
    (b"\xed\xa0\x80\xff", "byte 0xff in position 3: invalid start byte"),
    (b"\xed\xa0\x80\xed\xc0\x80", "byte 0xed in position 3: invalid continuation byte"),
    (b"\x41\xff", "byte 0xff in position 1: invalid start byte"),
    (b"\xed\xa0\x80\xc3", "byte 0xc3 in position 3: unexpected end of data"),
):
    for loads in loads_both(payload):
        assert raises(UnicodeDecodeError, loads) == f"'utf-8' codec can't decode {reason}"

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
for hex_arg, position in ((chr(0xDC80), 0), ("41" + chr(0xDC80) + "42", 2), ("41\u4e2d", 2)):
    reason = f"non-hexadecimal number found in fromhex() arg at position {position}"
    assert raises(ValueError, lambda a=hex_arg: bytes.fromhex(a)) == reason
    assert raises(ValueError, lambda a=hex_arg: bytearray.fromhex(a)) == reason
assert bytes.fromhex("41 42") == b"AB"

print("OK")
