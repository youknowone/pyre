# pyre-check: gate=1
"""PyPy's `_codecs_tw` Big5 and CP950 cjkcodecs engine."""

import _codecs_tw


for name, text in (("big5", "中文"), ("cp950", "中文€")):
    codec = _codecs_tw.getcodec(name)
    encoded, consumed = codec.encode(text)
    assert consumed == len(text), (name, consumed)
    assert codec.decode(encoded) == (text, len(encoded)), name


for name, text, encoded in (
    ("big5", "漢", bytes.fromhex("ba7e")),
    ("cp950", "漢€", bytes.fromhex("ba7ea3e1")),
):
    codec = _codecs_tw.getcodec(name)
    assert codec.encode(text) == (encoded, len(text)), name
    decoder = __import__("codecs").getincrementaldecoder(name)()
    pieces = [decoder.decode(bytes([byte]), False) for byte in encoded]
    pieces.append(decoder.decode(b"", True))
    assert "".join(pieces) == text, name


for name in ("big5", "cp950"):
    try:
        b"\x80\x80".decode(name)
    except UnicodeDecodeError as exc:
        assert (exc.start, exc.end) == (0, 1), (name, exc.start, exc.end)
    else:
        raise AssertionError(name)


print("OK")
