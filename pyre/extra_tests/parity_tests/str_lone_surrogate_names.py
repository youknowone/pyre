"""A `str` holding a lone surrogate survives being used as a name.

pyre stores every `str` as WTF-8 and decodes every operating-system-supplied
name with `surrogateescape`, so U+DC80..U+DCFF is reachable from ordinary code:
`sys.argv`, and on any filesystem that does not enforce UTF-8, every filename.
The `&str` accessor that most internals read through has no view of such a
value and used to abort the process rather than return one.

`func.__name__ = <surrogate>` was the reachable instance: it aborted with
`w_str_get_value: backing Wtf8Buf is not valid UTF-8 (lone surrogate)`. The
surrogate is built here from bytes rather than written as a literal, because
that is the only way it arrives in practice — and because a source file cannot
carry one.

Round-tripping is asserted, not just survival. The name is mirrored into a
UTF-8-only slot for the internal `&str` readers, and an escaped mirror would
satisfy "did not crash" while quietly returning a different string.
"""

import sys
import types

# b'\xff\xfe' is not valid UTF-8 in any position, so surrogateescape maps each
# byte to U+DCFF / U+DCFE — exactly what a filesystem name or an argv element
# produces on a host that does not enforce UTF-8.
SURR = b"bad\xff\xfename".decode("utf-8", "surrogateescape")


def check(cond, what):
    if not cond:
        raise AssertionError(what)


# The payload is the point of the test, so verify it before using it: a build
# that silently dropped the surrogates would otherwise pass everything below.
check(len(SURR) == 9, f"surrogate payload is {len(SURR)} code points, expected 9")
check(SURR.encode("utf-8", "surrogateescape") == b"bad\xff\xfename", "payload does not round-trip")
check(any(0xDC80 <= ord(c) <= 0xDCFF for c in SURR), "payload carries no lone surrogate")


def f():
    return 0


# ── the assignment that used to abort ────────────────────────────────────
f.__name__ = SURR
check(f.__name__ == SURR, f"__name__ came back as {f.__name__!r}, not the value set")
check(isinstance(repr(f), str), "repr of a surrogate-named function is not a str")

# The qualname slot is separate and must not have been clobbered by the name.
f.__qualname__ = SURR
check(f.__qualname__ == SURR, f"__qualname__ came back as {f.__qualname__!r}")

# ── the constructor arm takes the same name ──────────────────────────────
g = types.FunctionType(f.__code__, {}, SURR)
check(g.__name__ == SURR, f"FunctionType name came back as {g.__name__!r}")

# ── a plain function is unaffected ───────────────────────────────────────
def h():
    return 0


check(h.__name__ == "h", "an ordinary function lost its name")
h.__name__ = "renamed"
check(h.__name__ == "renamed", "an ordinary rename stopped working")

# ── the same value through the surfaces that read names ──────────────────
check(SURR in {SURR: 1}, "a surrogate-bearing key does not find itself in a dict")
check(hash(SURR) == hash(SURR), "hashing a surrogate-bearing str is not stable")
o = type("T", (), {})()
setattr(o, SURR, 7)
check(getattr(o, SURR) == 7, "attribute set/get by a surrogate-bearing name lost the value")
check(SURR in vars(o), "the surrogate-bearing attribute is missing from vars()")

# sys.argv is the other producer reachable without a filesystem; when this test
# is handed the payload it must arrive intact.
if len(sys.argv) > 1:
    check(sys.argv[1] == SURR, f"argv did not carry the payload: {sys.argv[1]!r}")

print("OK")
