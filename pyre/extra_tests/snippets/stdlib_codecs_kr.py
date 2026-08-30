# pyre-check: gate=1
"""PyPy's `_codecs_kr` cjkcodecs engine and its three codec families."""

import _codecs_kr
import codecs


for name, text in (
    ("euc_kr", "한국어와 漢字"),
    ("cp949", "똠방각하와 漢字"),
    ("johab", "한글과 漢字"),
):
    codec = _codecs_kr.getcodec(name)
    encoded, consumed = codec.encode(text)
    assert consumed == len(text), (name, consumed)
    assert codec.decode(encoded) == (text, len(encoded)), name


# PyPy's exact state-machine outputs, including the EUC-KR Annex 3 make-up
# sequence for syllables that only CP949 maps directly.
oracle_text = "가힣똠ㄱ漢字"
oracle_bytes = {
    "euc_kr": bytes.fromhex("b0a1a4d4a4bea4d3a4bea4d4a4a8a4c7a4b1a4a1f9d3edae"),
    "cp949": bytes.fromhex("b0a1c6528c63a4a1f9d3edae"),
    "johab": bytes.fromhex("8861d3bd99b18841f7d3f1ae"),
}
for name, encoded in oracle_bytes.items():
    codec = _codecs_kr.getcodec(name)
    assert codec.encode(oracle_text) == (encoded, len(oracle_text)), name
    decoder = codecs.getincrementaldecoder(name)()
    pieces = [decoder.decode(bytes([byte]), False) for byte in encoded]
    pieces.append(decoder.decode(b"", True))
    assert "".join(pieces) == oracle_text, name


# Pinned CPython 3.14 rejects the lead byte only for malformed double-byte
# candidates.  PyPy's checked-in codec returns a wider span, although the real
# pypy3 oracle exposes the same one-byte error as CPython.
for name, encoded in (
    ("euc_kr", b"\xff\xff"),
    ("cp949", b"\x81\x00"),
    ("johab", b"\x84\x00"),
):
    try:
        encoded.decode(name)
    except UnicodeDecodeError as exc:
        assert (exc.start, exc.end) == (0, 1), (name, exc.start, exc.end)
    else:
        raise AssertionError((name, encoded))


assert "A😀B".encode("cp949", "replace") == b"A?B"
assert "A😀B".encode("cp949", "ignore") == b"AB"
assert b"A\x81\x00B".decode("cp949", "replace") == "A�\x00B"
assert b"A\x81\x00B".decode("cp949", "ignore") == "A\x00B"


def encode_handler(exc):
    assert (exc.encoding, exc.start, exc.end, exc.reason) == (
        "cp949",
        1,
        2,
        "illegal multibyte sequence",
    )
    return (b"<x>", exc.end)


codecs.register_error("pyre_kr_encode", encode_handler)
assert "A😀B".encode("cp949", "pyre_kr_encode") == b"A<x>B"

decoder = codecs.getincrementaldecoder("cp949")()
assert decoder.decode(b"\x81", False) == ""
assert decoder.decode(b"A", True) == "갂"


print("OK")
