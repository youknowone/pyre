# pyre-check: pypy-diverges: pypy3's `sys.intern` states one sentence
# ("intern() argument must be string.") for both refusals, so the split this
# pins is not expressible there.
#
# CPython-suite gap: `test_sys.test_intern` interns strings and checks
# identity; `test_intern` in `test_unicode` covers the subclass case, and
# neither asserts what the non-string refusal says.
#
# parity-tests reason: `sys.intern` is declined twice over.  Its converter
# takes a `unicode`, so a `bytes` never reaches the body and is refused by the
# argument; a `str` subclass passes the converter and is refused by
# `sys_intern_impl`, which names only the type.  A runtime that states one
# sentence for both reports the subclass wording for an argument that was never
# a string.
import sys


def refusal(value):
    try:
        sys.intern(value)
    except TypeError as exc:
        return str(exc)
    raise AssertionError("intern accepted %r" % (value,))


class S(str):
    pass


assert refusal(b"x") == "intern() argument must be str, not bytes", refusal(b"x")
assert refusal(3) == "intern() argument must be str, not int", refusal(3)
assert refusal(S("a")) == "can't intern S", refusal(S("a"))
assert sys.intern("plain") == "plain"

print("OK")
