# pyre-check: gate=1
"""Incremental and sentinel behaviour of the backslash-escape codecs.

`unicode_escape_decode` and `raw_unicode_escape_decode` take a `final` flag.
While it is false the caller may still supply more input, so an escape running
off the end of the chunk is left unconsumed instead of reported; only a
sequence the chunk already decides reaches the error handler.  `charmap_decode`
reads U+FFFE as "undefined" however the table spells it.
"""
import codecs

BS = chr(92)


def check(got, want, what):
    assert got == want, "%s: %r != %r" % (what, got, want)


def raises(exc, fn, what):
    try:
        fn()
    except exc:
        return
    except BaseException as e:  # noqa: BLE001
        raise AssertionError("%s: raised %r, wanted %s" % (what, e, exc.__name__))
    raise AssertionError("%s: no %s" % (what, exc.__name__))


# --- unicode_escape: an escape running off the end is held back -------------
for tail in ["", "u", "u12", "u123", "x", "x4", "U", "U000", "N", "N{", "N{GREEK"]:
    data = (BS + tail).encode()
    check(codecs.unicode_escape_decode(data, "strict", False), ("", 0),
          "ue hold back %r" % data)
    raises(UnicodeDecodeError,
           lambda d=data: codecs.unicode_escape_decode(d, "strict", True),
           "ue final %r" % data)

# What precedes the held-back escape is still decoded and counted.
check(codecs.unicode_escape_decode(("a" + BS + "u12").encode(), "strict", False),
      ("a", 1), "ue prefix kept")
check(codecs.unicode_escape_decode(("a" + BS + "u1234" + BS + "x").encode(), "strict", False),
      ("a" + chr(0x1234), 7), "ue decoded prefix kept")

# A sequence the chunk decides is reported whether or not more input follows.
for tail in ["x4z", "xzz", "u12zz", "Nx"]:
    data = (BS + tail).encode()
    for is_final in (False, True):
        raises(UnicodeDecodeError,
               lambda d=data, f=is_final: codecs.unicode_escape_decode(d, "strict", f),
               "ue decided %r final=%s" % (data, is_final))

# A complete escape is unaffected by the flag.
for is_final in (False, True):
    check(codecs.unicode_escape_decode((BS + "u1234").encode(), "strict", is_final),
          (chr(0x1234), 6), "ue complete final=%s" % is_final)
    check(codecs.unicode_escape_decode((BS + "5").encode(), "strict", is_final),
          (chr(5), 2), "ue octal final=%s" % is_final)

# Holding back precedes the error handler, so a non-strict name changes nothing.
for handler in ("replace", "ignore", "backslashreplace"):
    check(codecs.unicode_escape_decode((BS + "u12").encode(), handler, False), ("", 0),
          "ue hold back beats %s" % handler)

# `final` defaults to true.
raises(UnicodeDecodeError, lambda: codecs.unicode_escape_decode((BS + "u12").encode()),
       "ue default final")


# --- raw_unicode_escape: only \uXXXX and \UXXXXXXXX are escapes -------------
for tail in ["", "u", "u1", "u12", "u123", "U", "U0001"]:
    data = (BS + tail).encode()
    check(codecs.raw_unicode_escape_decode(data, "strict", False), ("", 0),
          "rue hold back %r" % data)

# A lone backslash is literal once no more input can arrive.
check(codecs.raw_unicode_escape_decode(BS.encode(), "strict", True), (BS, 1),
      "rue lone backslash")
# Bytes that cannot introduce an escape are literal even mid-stream.
for tail in ["x", "z", "N{A"]:
    data = (BS + tail).encode()
    want = (BS + tail, len(data))
    for is_final in (False, True):
        check(codecs.raw_unicode_escape_decode(data, "strict", is_final), want,
              "rue literal %r final=%s" % (data, is_final))

# Four bytes that are not hex are decided, so they report.
raises(UnicodeDecodeError,
       lambda: codecs.raw_unicode_escape_decode((BS + "u12zz").encode(), "strict", False),
       "rue decided")
raises(UnicodeDecodeError,
       lambda: codecs.raw_unicode_escape_decode((BS + "u12").encode()),
       "rue default final")

check(codecs.raw_unicode_escape_decode(("a" + BS + "u12").encode(), "strict", False),
      ("a", 1), "rue prefix kept")

# str and any buffer producer are accepted, not just bytes.
check(codecs.raw_unicode_escape_decode(BS + "u1234"), (chr(0x1234), 6), "rue str input")
check(codecs.raw_unicode_escape_decode(memoryview((BS + "u1234").encode())),
      (chr(0x1234), 6), "rue memoryview input")
check(codecs.unicode_escape_decode(memoryview((BS + "u1234").encode())),
      (chr(0x1234), 6), "ue memoryview input")


# --- charmap_decode: U+FFFE means undefined, as an int or as a str ----------
a, b = ord("a"), ord("b")
data = bytes([0, 1, 2])
for table in ({0: a, 1: b, 2: 0xFFFE}, {0: a, 1: b, 2: chr(0xFFFE)}, {0: a, 1: b}):
    raises(UnicodeDecodeError,
           lambda t=table: codecs.charmap_decode(data, "strict", t),
           "charmap undefined %r" % (table,))
    check(codecs.charmap_decode(data, "replace", table), ("ab" + chr(0xFFFD), 3),
          "charmap replace %r" % (table,))
    check(codecs.charmap_decode(data, "backslashreplace", table),
          ("ab" + BS + "x02", 3), "charmap backslashreplace %r" % (table,))

# A code point outside the Unicode range is a TypeError, not an undefined byte.
raises(TypeError,
       lambda: codecs.charmap_decode(data, "strict", {0: 0x110000, 1: b, 2: a}),
       "charmap out of range")


# --- escape_encode quotes an apostrophe, not a double quote -----------------
check(codecs.escape_encode(b"a'b" + chr(34).encode()), (b"a" + BS.encode() + b"'b" + chr(34).encode(), 4),
      "escape_encode apostrophe")


# --- \N{NAME} resolves through the character database ----------------------
def esc(name):
    return codecs.unicode_escape_decode((BS + "N{" + name + "}").encode())


check(esc("GREEK SMALL LETTER ALPHA"), (chr(0x3B1), 28), "N named")
check(esc("greek small letter alpha"), (chr(0x3B1), 28), "N name is case-blind")
check(esc("NUL"), (chr(0), 7), "N alias")
check(esc("LATIN SMALL LETTER A"), ("a", 24), "N ascii name")
check(esc("CJK UNIFIED IDEOGRAPH-4E00"), (chr(0x4E00), 30), "N generated name")

# Only a name that names ONE character resolves; a named sequence does not,
# even though `unicodedata.lookup` answers for it.
raises(UnicodeDecodeError, lambda: esc("KEYCAP DIGIT ZERO"), "N named sequence")
raises(UnicodeDecodeError, lambda: esc("NOTANAME"), "N unknown name")

# An empty, unterminated, or brace-less name is malformed rather than unknown,
# and each spelling reports its own span.
SPANS = [
    ("N{}", "malformed", 0, 3),
    ("N{ABC", "malformed", 0, 6),
    ("N", "malformed", 0, 2),
    ("Nx", "malformed", 0, 2),
    ("N{NOTANAME}", "unknown Unicode character name", 0, 12),
]
for tail, reason, start, end in SPANS:
    data = (BS + tail).encode()
    try:
        codecs.unicode_escape_decode(data)
    except UnicodeDecodeError as e:
        assert reason in e.reason, (tail, e.reason)
        assert (e.start, e.end) == (start, end), (tail, e.start, e.end)
    else:
        raise AssertionError("no UnicodeDecodeError for %r" % data)

print("OK")
