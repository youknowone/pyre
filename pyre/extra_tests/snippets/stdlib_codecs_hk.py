# pyre-check: gate=1
"""PyPy's `_codecs_hk` BIG5-HKSCS state machine."""

import _codecs_hk
import codecs


codec = _codecs_hk.getcodec("big5hkscs")
for text, encoded in (
    ("漢", bytes.fromhex("ba7e")),
    ("Ê", bytes.fromhex("8866")),
    ("Ê\u0304", bytes.fromhex("8862")),
    ("Ê\u030c", bytes.fromhex("8864")),
    ("ê\u0304", bytes.fromhex("88a3")),
    ("ê\u030c", bytes.fromhex("88a5")),
):
    assert codec.encode(text) == (encoded, len(text)), text
    assert codec.decode(encoded) == (text, len(encoded)), text


encoder = codecs.getincrementalencoder("big5hkscs")()
assert encoder.encode("Ê", False) == b""
assert encoder.encode("\u0304", True) == bytes.fromhex("8862")

try:
    b"\x81\x00".decode("big5hkscs")
except UnicodeDecodeError as exc:
    assert (exc.start, exc.end) == (0, 1)
else:
    raise AssertionError("malformed BIG5-HKSCS candidate was accepted")


print("OK")
