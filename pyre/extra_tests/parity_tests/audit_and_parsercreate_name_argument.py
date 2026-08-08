"""`sys.audit` and `pyexpat.ParserCreate` check the name they are handed.

`sys.audit` was registered as `|_| Ok(w_none())` — a no-op that accepted
anything, so `sys.audit()` with no event at all, with an `int`, or by keyword
all returned None.  22 stdlib modules call it (`os.walk`, `glob.glob`,
`pickle.find_class`, `subprocess`, ...), so a bad event name reached none of the
checks that upstream's `@unwrap_spec(event="text")` performs.

`ParserCreate` did check that `encoding` is `str` or `None`, but stored it
without asking whether it has a UTF-8 spelling — the sibling
`namespace_separator` arm did ask.  A `str` holding a lone surrogate therefore
built a parser whose stored encoding no `&str` reader can see.  The same arm
also spelled its type name as a literal `int`, so every other type was reported
as `int`.

No audit hook can be installed yet (`sys.addaudithook` stores nothing), so
`sys.audit` still does nothing past the checks below; the hook mechanism is a
separate gap.  The surrogate is built from bytes rather than written as a
literal because that is how it arrives in practice and because a source file
cannot carry one.

Only one surrogate is used for the messages that are compared verbatim: a run
of adjacent unencodable code points is reported by CPython as one error with a
range (`characters in position 3-4`) and by pyre as the first one alone, which
is a separate divergence in the encoder rather than in these entry points.
"""

import sys

import pyexpat

# b'\xff' has no UTF-8 spelling in any position, so surrogateescape maps it to
# U+DCFF — what a filesystem name or an argv element yields on a host that does
# not enforce UTF-8.
SURR = b"bad\xffname".decode("utf-8", "surrogateescape")
ENCODE_ERROR = (
    "'utf-8' codec can't encode character '\\udcff' in position 3: "
    "surrogates not allowed"
)

ERRORS = []


def check(cond, what):
    if not cond:
        ERRORS.append(what)


def raises(what, exc, expected, fn):
    """Assert fn() raises `exc` whose message is exactly `expected`."""
    try:
        fn()
    except exc as e:
        check(
            str(e) == expected,
            f"{what}: got {str(e)!r}, expected {expected!r}",
        )
        return
    except BaseException as e:
        ERRORS.append(
            f"{what}: raised {type(e).__name__}({e!r}), expected {exc.__name__}"
        )
        return
    ERRORS.append(f"{what}: no exception, expected {exc.__name__}")


# The payload is the point of the test, so verify it before using it.
check(len(SURR) == 8, f"surrogate payload is {len(SURR)} code points, expected 8")
check(SURR.encode("utf-8", "surrogateescape") == b"bad\xffname", "payload lost its byte")


class Odd:
    pass


# ── sys.audit: the event name is a str that must encode ───────────────────
raises(
    "audit with no event",
    TypeError,
    "audit expected at least 1 argument, got 0",
    lambda: sys.audit(),
)
for value, spelling in (
    (123, "int"),
    (1.5, "float"),
    (None, "None"),
    (b"x", "bytes"),
    (bytearray(b"x"), "bytearray"),
    ((), "tuple"),
    (Odd(), "Odd"),
):
    raises(
        f"audit({spelling})",
        TypeError,
        f"audit() argument 1 must be str, not {spelling}",
        lambda value=value: sys.audit(value),
    )
raises(
    "audit with a surrogate event",
    UnicodeEncodeError,
    ENCODE_ERROR,
    lambda: sys.audit(SURR),
)
# Every parameter is positional-only, so the event cannot be named.
raises(
    "audit by keyword",
    TypeError,
    "sys.audit() takes no keyword arguments",
    lambda: sys.audit(event="x"),
)
# ...and the accepting calls still accept, so the checks above are not passing
# because audit stopped working.
check(sys.audit("pyre.test") is None, "audit rejected a plain event name")
check(sys.audit("pyre.test", 1, 2) is None, "audit rejected trailing parameters")


class SubStr(str):
    pass


check(sys.audit(SubStr("pyre.test")) is None, "audit rejected a str subclass")


# ── pyexpat.ParserCreate: both str-or-None parameters report their type ────
for value, spelling in ((123, "int"), (1.5, "float"), (b"x", "bytes")):
    raises(
        f"ParserCreate(encoding={spelling})",
        TypeError,
        f"ParserCreate() argument 'encoding' must be str or None, not {spelling}",
        lambda value=value: pyexpat.ParserCreate(value),
    )
    raises(
        f"ParserCreate(namespace_separator={spelling})",
        TypeError,
        f"ParserCreate() argument 'namespace_separator' must be str or None, "
        f"not {spelling}",
        lambda value=value: pyexpat.ParserCreate(None, value),
    )

raises(
    "ParserCreate with a surrogate encoding",
    UnicodeEncodeError,
    ENCODE_ERROR,
    lambda: pyexpat.ParserCreate(SURR),
)
raises(
    "ParserCreate with a surrogate separator",
    UnicodeEncodeError,
    ENCODE_ERROR,
    lambda: pyexpat.ParserCreate(None, SURR),
)
# The encoding check runs before the separator's length check, so a too-long
# separator carrying a surrogate reports the encoding, not the length.
raises(
    "ParserCreate with a long surrogate separator",
    UnicodeEncodeError,
    ENCODE_ERROR,
    lambda: pyexpat.ParserCreate(None, SURR + "x"),
)
raises(
    "ParserCreate with a two-character separator",
    ValueError,
    "namespace_separator must be at most one character, omitted, or None",
    lambda: pyexpat.ParserCreate(None, "ab"),
)

# The accepting calls, so the checks above are not passing because
# ParserCreate stopped building parsers.
check(
    type(pyexpat.ParserCreate()).__name__ == "xmlparser",
    "ParserCreate() no longer builds a parser",
)
check(
    type(pyexpat.ParserCreate("utf-8")).__name__ == "xmlparser",
    "ParserCreate('utf-8') no longer builds a parser",
)
check(
    type(pyexpat.ParserCreate(encoding="utf-8")).__name__ == "xmlparser",
    "ParserCreate(encoding=) no longer binds",
)
check(
    type(pyexpat.ParserCreate(None, ":")).__name__ == "xmlparser",
    "ParserCreate with a separator no longer builds a parser",
)
check(
    type(pyexpat.ParserCreate(SubStr("utf-8"))).__name__ == "xmlparser",
    "ParserCreate rejected a str subclass encoding",
)

# ── ctypes.CDLL: the library name reaches dlopen in filesystem units ──────
# Filed alongside the two above as a third disagreement, and already correct:
# `_ctypes.dlopen` takes the name through `fsencode`, so a surrogate escape
# folds back to the byte it stands for and dlopen is the thing that fails.
# Pinned because the earlier reading came from a binary built before that
# routing landed. A surrogate that is *not* an escape (U+D800 is outside
# U+DC80..U+DCFF) has no byte to fold to, so it is refused before the call —
# which is what the second row separates from the first.
try:
    import ctypes
except ImportError:
    ctypes = None
if ctypes is not None:

    def raises_class(what, exc, fn):
        """Assert fn() raises `exc`; the message is the host's dlopen text."""
        try:
            fn()
        except exc:
            return
        except BaseException as e:
            ERRORS.append(
                f"{what}: raised {type(e).__name__}, expected {exc.__name__}"
            )
            return
        ERRORS.append(f"{what}: no exception, expected {exc.__name__}")

    raises_class(
        "CDLL with a surrogate-escaped name",
        OSError,
        lambda: ctypes.CDLL(SURR),
    )
    raises_class(
        "CDLL with a non-escape surrogate",
        UnicodeEncodeError,
        lambda: ctypes.CDLL("bad\ud800name"),
    )
    raises_class(
        "CDLL with a missing library",
        OSError,
        lambda: ctypes.CDLL("nosuchlib_pyre_parity_test"),
    )

if ERRORS:
    for e in ERRORS:
        sys.stderr.write(f"FAIL: {e}\n".encode("utf-8", "backslashreplace").decode())
    raise AssertionError(f"{len(ERRORS)} divergence(s)")

print("OK")
