# pyre-check: pypy-diverges: pins the one-byte span of a surrogate the surrogatepass decoders cannot complete; pypy3 reports (0, 2)
# CPython-suite gap: test_codeccallbacks exercises surrogatepass round-trips
# but never the span of a truncated surrogate, and neither entry point is
# compared against the other anywhere.
# parity-tests reason: the span is produced by pyre's own state machine, whose
# `allow_surrogates` arm has no counterpart in `unicode_decode_utf8`.

"""A surrogate the `surrogatepass` decoders cannot complete spans one byte."""

import codecs
import _codecs


def span(fn):
    try:
        fn()
    except UnicodeDecodeError as e:
        return e.start, e.end, e.reason
    raise AssertionError("UnicodeDecodeError not raised")


# `str_decode_utf8` defaults `allow_surrogates` off, so the state machine stops
# at the second byte and `surrogatepass_errors` is what decodes a complete
# sequence.
assert span(lambda: b"\xed\xa0".decode("utf-8", "surrogatepass")) == (
    0, 1, "invalid continuation byte")
assert span(lambda: b"\xed\xa0\x41".decode("utf-8", "surrogatepass")) == (
    0, 1, "invalid continuation byte")
assert span(lambda: b"\xed".decode("utf-8", "surrogatepass")) == (
    0, 1, "unexpected end of data")
assert span(lambda: b"\xe0\xa0".decode("utf-8", "surrogatepass")) == (
    0, 2, "unexpected end of data")

# `interp_codecs.utf_8_decode` turns `allow_surrogates` on, which admits a
# whole `ED A0..BF 80..BF` and nothing less: a pair that does not complete
# reports the byte the allowance was suspending judgement on, not the pair.
for final in (True, False):
    assert span(lambda f=final: _codecs.utf_8_decode(b"\xed\xa0\x41", "surrogatepass", f)) == (
        0, 1, "invalid continuation byte")
    assert span(lambda f=final: _codecs.utf_8_decode(b"\xed\xa0\xff", "surrogatepass", f)) == (
        0, 1, "invalid continuation byte")
    assert span(lambda f=final: _codecs.utf_8_decode(b"\x41\xed\xa0\x42", "surrogatepass", f)) == (
        1, 2, "invalid continuation byte")
assert span(lambda: _codecs.utf_8_decode(b"\xed\xa0", "surrogatepass", True)) == (
    0, 1, "invalid continuation byte")

# A lead pair that is not a surrogate keeps the two-byte span, and so does a
# four-byte sequence, so the arm above is the only one that narrowed.
for data, expected in (
    (b"\xe4\xb8", (0, 2, "unexpected end of data")),
    (b"\xe4\xb8\x41", (0, 2, "invalid continuation byte")),
    (b"\xed\x9f\x41", (0, 2, "invalid continuation byte")),
    (b"\xf0\x9f\x98", (0, 3, "unexpected end of data")),
    (b"\xf0\x9f\x98\x41", (0, 3, "invalid continuation byte")),
    (b"\xe0\x80", (0, 1, "invalid continuation byte")),
    (b"\xf0\x8f", (0, 1, "invalid continuation byte")),
):
    assert span(lambda d=data: _codecs.utf_8_decode(d, "surrogatepass", True)) == expected, data

# A truncated pair at the end of a non-final chunk is still retained rather
# than rejected -- the narrowing above applies only once the chunk is final.
assert _codecs.utf_8_decode(b"\xed\xa0", "surrogatepass", False) == ("", 0)
decoder = codecs.getincrementaldecoder("utf-8")("surrogatepass")
assert decoder.decode(b"\xed\xa0", False) == ""
assert decoder.decode(b"\x80", True) == "\ud800"

# The allowance belongs to `surrogatepass` alone: every other handler sees a
# complete encoded surrogate as the bad continuation byte it is, and answers
# in its own way rather than decoding it.
assert span(lambda: _codecs.utf_8_decode(b"\xed\xa0\x80", "strict", True)) == (
    0, 1, "invalid continuation byte")
for errors, expected in (
    ("replace", ("�" * 3, 3)),
    ("ignore", ("", 3)),
    ("backslashreplace", ("\\xed\\xa0\\x80", 3)),
    ("surrogateescape", ("\udced\udca0\udc80", 3)),
    ("surrogatepass", ("\ud800", 3)),
):
    assert _codecs.utf_8_decode(b"\xed\xa0\x80", errors, True) == expected, errors

# Every complete sequence still round-trips through both entry points.
for subject in ("\ud800", "\udfff", "\U00010000", "a\udc80b", "\ud800" * 50,
                "abc", "\xe9中\U00010000", ""):
    encoded = subject.encode("utf-8", "surrogatepass")
    assert encoded.decode("utf-8", "surrogatepass") == subject
    assert _codecs.utf_8_decode(encoded, "surrogatepass", True) == (subject, len(encoded))
    assert str(encoded, "utf-8", "surrogatepass") == subject

print("OK")
