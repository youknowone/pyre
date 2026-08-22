# CPython-suite gap: test_codeccallbacks exercises surrogatepass round-trips
# but never the span of a truncated surrogate, and the two entry points that
# disagree about it are not compared anywhere.
# parity-tests reason: `_codecs.utf_8_decode`'s answer follows PyPy, not
# CPython, so only the `bytes.decode` half can be asserted against the oracle.

"""`bytes.decode` and `_codecs.utf_8_decode` split on `allow_surrogates`."""

import codecs
import sys
import _codecs


def span(fn):
    try:
        fn()
    except UnicodeDecodeError as e:
        return e.start, e.end, e.reason
    raise AssertionError("UnicodeDecodeError not raised")


# `str_decode_utf8` defaults `allow_surrogates` off, so the state machine stops
# at the second byte and `surrogatepass_errors` is what decodes a complete
# sequence.  CPython agrees here.
assert span(lambda: b"\xed\xa0".decode("utf-8", "surrogatepass")) == (
    0, 1, "invalid continuation byte")
assert span(lambda: b"\xed\xa0\x41".decode("utf-8", "surrogatepass")) == (
    0, 1, "invalid continuation byte")
assert span(lambda: b"\xed".decode("utf-8", "surrogatepass")) == (
    0, 1, "unexpected end of data")
assert span(lambda: b"\xe0\xa0".decode("utf-8", "surrogatepass")) == (
    0, 2, "unexpected end of data")

# Every complete sequence still round-trips through both entry points.
for subject in ("\ud800", "\udfff", "\U00010000", "a\udc80b", "\ud800" * 50,
                "abc", "\xe9中\U00010000", ""):
    encoded = subject.encode("utf-8", "surrogatepass")
    assert encoded.decode("utf-8", "surrogatepass") == subject
    assert _codecs.utf_8_decode(encoded, "surrogatepass", True) == (subject, len(encoded))
    assert str(encoded, "utf-8", "surrogatepass") == subject

# A surrogate split across incremental chunks is retained, not rejected.
decoder = codecs.getincrementaldecoder("utf-8")("surrogatepass")
assert decoder.decode(b"\xed\xa0", False) == ""
assert decoder.decode(b"\x80", True) == "\ud800"

if sys.implementation.name != "cpython":
    # `interp_codecs.utf_8_decode` turns `allow_surrogates` on, so the same
    # two bytes are an incomplete sequence rather than a bad continuation.
    assert span(lambda: _codecs.utf_8_decode(b"\xed\xa0", "surrogatepass", True)) == (
        0, 2, "unexpected end of data")
    assert span(lambda: _codecs.utf_8_decode(b"\xed\xa0\x41", "surrogatepass", True)) == (
        0, 2, "invalid continuation byte")

print("OK")
