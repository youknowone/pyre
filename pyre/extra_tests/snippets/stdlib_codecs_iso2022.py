# pyre-check: gate=1
"""PyPy's stateful `_codecs_iso2022` engine, ported line by line to Rust."""

import codecs


# Real PyPy 7.3.20 oracle vectors.  CPython 3.14.2 produces the same bytes.
vectors = {
    "iso2022_kr": ("한국어", b"\x1b$)C\x0eGQ19>n\x0f"),
    "iso2022_jp": ("日本語", b"\x1b$BF|K\\8l\x1b(B"),
    "iso2022_jp_1": ("日本語", b"\x1b$BF|K\\8l\x1b(B"),
    "iso2022_jp_2": ("한국어", b"\x1b$(CGQ19>n\x1b(B"),
    "iso2022_jp_2004": ("か\u309a\U0002000b", b"\x1b$(Q$w.\x22\x1b(B"),
    "iso2022_jp_3": ("か\u309a\U0002000b", b"\x1b$(O$w.\x22\x1b(B"),
    "iso2022_jp_ext": ("日本語", b"\x1b$BF|K\\8l\x1b(B"),
}
for encoding, (text, encoded) in vectors.items():
    assert text.encode(encoding) == encoded, (encoding, text.encode(encoding).hex())
    assert encoded.decode(encoding) == text


for encoding in vectors:
    encoder = codecs.getincrementalencoder(encoding)()
    chunks = [encoder.encode("日", False), encoder.encode("本", False), encoder.encode("", True)]
    assert b"".join(chunks) == "日本".encode(encoding), (encoding, chunks)

    decoder = codecs.getincrementaldecoder(encoding)()
    decoded = []
    for byte in b"".join(chunks):
        decoded.append(decoder.decode(bytes([byte]), False))
    decoded.append(decoder.decode(b"", True))
    assert "".join(decoded) == "日本", (encoding, decoded)


for encoding, mark in (("iso2022_jp_2004", b"Q"), ("iso2022_jp_3", b"O")):
    # Unlike the EUC/Shift-JIS engines, PyPy's ISO-2022 engine does not apply
    # its full-width-tilde compatibility override to JIS row 0x22/0x32.
    assert (b"\x1b$(" + mark + b"\x22\x32\x1b(B").decode(encoding) == "~"
    assert ("か\0").encode(encoding).endswith(b"\0")


for encoding in ("iso2022_jp", "iso2022_jp_2", "iso2022_jp_2004"):
    for data, span in (
        (b"\x1b", (0, 1)),
        (b"\x1b$", (0, 2)),
        (b"\x1b$(", (0, 3)),
        (b"\x1b$(Z", (0, 4)),
        (b"\x1b$B\x21\x00", (3, 5)),
        (b"\x80", (0, 1)),
    ):
        try:
            data.decode(encoding)
        except UnicodeDecodeError as error:
            assert (error.start, error.end) == span, (encoding, data, error)
        else:
            raise AssertionError((encoding, data))
    assert b"\x1bXabcZ".decode(encoding) == "\x1bXabcZ"
