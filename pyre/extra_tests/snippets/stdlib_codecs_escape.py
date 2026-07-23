"""PyPy _codecs.escape_decode parity used by protocol-0 pickle."""

import _codecs
import codecs


cases = {
    b"a\\n\\\\b\\x00c\\td": b"a\n\\b\x00c\td",
    b"\\077": b"?",
    b"\\100": b"@",
    b"\\253": bytes([0o253]),
    b"\\312": bytes([0o312]),
    b"\\400": b"\0",
    b"\\9": b"\\9",
    b"\\01": b"\x01",
    b"\\0f": b"\0f",
    b"\\08": b"\08",
}
for encoded, expected in cases.items():
    decoded, consumed = _codecs.escape_decode(encoded)
    assert decoded == expected, (encoded, decoded, expected)
    assert consumed == len(encoded)

raw = b"a\n\\b\x00c\td\xe5"
assert _codecs.escape_encode(raw) == (b"a\\n\\\\b\\x00c\\td\\xe5", len(raw))

assert codecs.escape_decode(b"a\\x00\\n") == (b"a\x00\n", 7)
assert _codecs.escape_decode(b"[\\x]\\x", "ignore") == (b"[]", 6)
assert _codecs.escape_decode(b"[\\x]\\x", "replace") == (b"[?]?", 6)

for malformed in (b"\\x", b"[\\x]", b"\\x0", b"[\\x0]", b"trailing\\"):
    try:
        _codecs.escape_decode(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError(malformed)
