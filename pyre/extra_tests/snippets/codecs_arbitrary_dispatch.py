# pyre-check: gate=1
"""`codecs.encode` / `codecs.decode` reach the registry, not the text model.

`PyCodec_Encode` places no restriction on either side of the codec: it looks the
name up without the text-encoding test and answers with whatever the registered
coder's first result element is.  The `errors` argument is likewise not
invented -- omitting it calls the coder with one argument.  Every `_codecs`
entry point spells the strict handler as `None` too.
"""
import codecs


def raises(exc, pattern, fn, what):
    try:
        fn()
    except exc as e:
        assert pattern in str(e), "%s: %r lacks %r" % (what, str(e), pattern)
        return
    except BaseException as e:  # noqa: BLE001
        raise AssertionError("%s: raised %r, wanted %s" % (what, e, exc.__name__))
    raise AssertionError("%s: no %s" % (what, exc.__name__))


CALLS = []


def make(name, encoder, decoder):
    def search(query, _name=name, _e=encoder, _d=decoder):
        if query == _name:
            return codecs.CodecInfo(_e, _d, name=_name)
        return None

    codecs.register(search)


def enc(*args, **kwds):
    CALLS.append(("encode", len(args)))
    return "not bytes!", 0


def dec(*args, **kwds):
    CALLS.append(("decode", len(args)))
    return b"not str!", 0


make("pyre_arbitrary", enc, dec)

# The output type is not policed, and the input is passed on as it stands.
for obj in (None, object(), "text", b"bytes", 17):
    assert codecs.encode(obj, "pyre_arbitrary") == "not bytes!", obj
    assert codecs.decode(obj, "pyre_arbitrary") == b"not str!", obj

# An omitted `errors` is not invented; supplying one adds the argument.
CALLS.clear()
codecs.encode(None, "pyre_arbitrary")
codecs.decode(None, "pyre_arbitrary")
assert CALLS == [("encode", 1), ("decode", 1)], CALLS
CALLS.clear()
codecs.encode(None, "pyre_arbitrary", "replace")
codecs.decode(None, "pyre_arbitrary", "replace")
assert CALLS == [("encode", 2), ("decode", 2)], CALLS
# `None` is a value, not an omission: the coder is not asked to accept it.
raises(TypeError, "encode() argument 'errors' must be str, not None",
       lambda: codecs.encode(None, "pyre_arbitrary", None), "errors=None")

# The text model still refuses what the arbitrary path allows.
raises(TypeError, "instead of 'bytes'", lambda: "s".encode("pyre_arbitrary"),
       "str.encode still checks")
raises(TypeError, "instead of 'str'", lambda: b"b".decode("pyre_arbitrary"),
       "bytes.decode still checks")

# A coder that does not answer with a 2-tuple is reported, whatever it returns.
for i, bad in enumerate([None, ("x",), ("x", 0, 1), "ab", ["x", 0]]):
    name = "pyre_badcoder%d" % i
    make(name, lambda *a, _b=bad, **k: _b, dec)
    raises(TypeError, "encoder must return a tuple (object, integer)",
           lambda n=name: codecs.encode(None, n), "bad encoder result %r" % (bad,))

make("pyre_baddecoder", enc, lambda *a, **k: None)
raises(TypeError, "decoder must return a tuple (object,integer)",
       lambda: codecs.decode(None, "pyre_baddecoder"), "bad decoder result")

# Argument reporting.
raises(LookupError, "unknown encoding: pyre_no_such",
       lambda: codecs.encode(None, "pyre_no_such"), "unknown encode")
raises(LookupError, "unknown encoding: pyre_no_such",
       lambda: codecs.decode(None, "pyre_no_such"), "unknown decode")
raises(TypeError, "encode() argument 'encoding' must be str, not int",
       lambda: codecs.encode(None, 5), "encoding type")
raises(TypeError, "decode() argument 'encoding' must be str, not int",
       lambda: codecs.decode(None, 5), "decoding type")
raises(TypeError, "encode() argument 'errors' must be str, not int",
       lambda: codecs.encode(None, "pyre_arbitrary", 5), "errors type")

# The ordinary codecs keep working, by keyword as well as by position.
assert codecs.encode("abc") == b"abc"
assert codecs.decode(b"abc") == "abc"
assert codecs.encode("\xe4\xf6\xfc", "latin-1") == b"\xe4\xf6\xfc"
assert codecs.decode(b"\xe4\xf6\xfc", "latin-1") == "\xe4\xf6\xfc"
assert codecs.encode(obj="\xe4\xf6\xfc", encoding="latin-1") == b"\xe4\xf6\xfc"
assert codecs.decode(obj=b"[\xff]", encoding="ascii", errors="ignore") == "[]"
assert codecs.encode("[\xff]", "ascii", errors="ignore") == b"[]"

# A text codec still reports a wrong-typed input itself.
raises(TypeError, "must be str", lambda: codecs.encode(b"x", "utf-8"),
       "bytes to a text encoder")


# --- `errors=None` names the strict handler at every entry point ------------
NONE_CASES = [
    ("unicode_escape_decode", codecs.unicode_escape_decode, (b"a",), ("a", 1)),
    ("raw_unicode_escape_decode", codecs.raw_unicode_escape_decode, (b"a",), ("a", 1)),
    ("escape_decode", codecs.escape_decode, (b"a",), (b"a", 1)),
    ("charmap_decode", codecs.charmap_decode, (b"a",), ("a", 1)),
    ("utf_8_decode", codecs.utf_8_decode, (b"a",), ("a", 1)),
    ("ascii_decode", codecs.ascii_decode, (b"a",), ("a", 1)),
    ("latin_1_decode", codecs.latin_1_decode, (b"a",), ("a", 1)),
    ("charmap_encode", codecs.charmap_encode, ("a",), (b"a", 1)),
    ("ascii_encode", codecs.ascii_encode, ("a",), (b"a", 1)),
    ("utf_8_encode", codecs.utf_8_encode, ("a",), (b"a", 1)),
]
for label, fn, args, want in NONE_CASES:
    got = fn(*args, None)
    assert got == want, "%s(errors=None): %r != %r" % (label, got, want)

# A name that is neither a str nor None is still refused.
for label, fn, args, _want in NONE_CASES:
    raises(TypeError, "", lambda f=fn, a=args: f(*a, 5), "%s(errors=5)" % label)

print("OK")
