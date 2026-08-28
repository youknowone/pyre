# pyre-check: gate=1
"""PyPy's `_codecs_jp` state machines, ported without the C engine."""

import _codecs_jp
import codecs


cases = (
    ("shift_jis", "日本語¥‾＼", bytes.fromhex("93fa967b8cea5c7e815f"), "日本語\\~＼"),
    ("cp932", "日本語\uf8f0\ue000", bytes.fromhex("93fa967b8ceaa0f040"), "日本語\uf8f0\ue000"),
    ("euc_jp", "日本語¥‾＼", bytes.fromhex("c6fccbdcb8ec5c7ea1c0"), "日本語\\~＼"),
    ("shift_jis_2004", "か\u309a\U0002000b", bytes.fromhex("82f587a0"), "か\u309a\U0002000b"),
    ("euc_jis_2004", "か\u309a\U0002000b", bytes.fromhex("a4f7aea2"), "か\u309a\U0002000b"),
    ("shift_jisx0213", "か\u309a\u9b1d", bytes.fromhex("82f5fc5a"), "か\u309a\u9b1d"),
    ("euc_jisx0213", "か\u309a\u9b1d", bytes.fromhex("a4f78ffdbb"), "か\u309a\u9b1d"),
)
for name, text, encoded, decoded in cases:
    codec = _codecs_jp.getcodec(name)
    assert codec.encode(text) == (encoded, len(text)), name
    assert codec.decode(encoded) == (decoded, len(encoded)), name


for name in ("shift_jis_2004", "euc_jis_2004"):
    encoder = codecs.getincrementalencoder(name)()
    assert encoder.encode("か", False) == b""
    assert encoder.encode("\u309a", True) == (
        bytes.fromhex("82f5") if name == "shift_jis_2004" else bytes.fromhex("a4f7")
    )


for name in ("shift_jis_2004", "shift_jisx0213"):
    assert b"\x81\x5f".decode(name) == "\\"
for name in ("euc_jis_2004", "euc_jisx0213", "shift_jis_2004", "shift_jisx0213"):
    assert ("フルーツ\0").encode(name).endswith(b"\0")


for name, encoded in (
    ("shift_jis", b"\x81\x00"),
    ("cp932", b"\x81\x00"),
    ("euc_jp", b"\x8f\xa1\x00"),
    ("shift_jis_2004", b"\x81\x00"),
    ("euc_jis_2004", b"\x8f\xa1\x00"),
):
    try:
        encoded.decode(name)
    except UnicodeDecodeError as exc:
        assert (exc.start, exc.end, exc.reason) == (
            0,
            1,
            "illegal multibyte sequence",
        ), name
    else:
        raise AssertionError(name)


print("OK")
