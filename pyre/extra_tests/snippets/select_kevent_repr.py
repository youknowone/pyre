# pyre-check: gate=1
# pyre-check: platforms=darwin
# pyre-check: pypy-diverges: `interp_kqueue.py W_Kevent` registers no
# `__repr__`, so pypy3 prints the default `<select.kevent object at 0x...>`.
# CPython-suite gap: test_kqueue does not import, so nothing in the suite reprs
# a kevent.
# parity-tests reason: pyre's `select.kevent` had no `__repr__` at all, so the
# six fields a caller sets were invisible in any log line or test failure that
# printed the object.

import select

default = select.kevent(0)
assert repr(default) == (
    "<select.kevent ident=0 filter=-1 flags=0x1 fflags=0x0 data=0x0 udata=0x0>"
), repr(default)

full = select.kevent(
    ident=7,
    filter=select.KQ_FILTER_WRITE,
    flags=select.KQ_EV_ADD | select.KQ_EV_ENABLE,
    fflags=5,
    data=12345,
    udata=0xDEADBEEF,
)
assert repr(full) == (
    "<select.kevent ident=7 filter=-2 flags=0x5 fflags=0x5 data=0x3039"
    " udata=0xdeadbeef>"
), repr(full)

# `data` is spelled as a `long long`, so a negative one reads as its two's
# complement rather than with a sign.
negative = select.kevent(0, data=-1)
assert "data=0xffffffffffffffff" in repr(negative), repr(negative)

print("OK")
