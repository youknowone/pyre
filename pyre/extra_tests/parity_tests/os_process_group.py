"""os.getpgrp, os.getpgid and os.ctermid answer the host, not None.

All three sat in the stub list answering `None`, which is a number a caller
cannot tell from a group id and a name it cannot tell from a terminal. The
checks here are the ones a `None` fails: that the two group calls agree about
the process they are both asked about, and that the terminal name is a path.
"""

import os
import sys


def check(cond, what):
    if not cond:
        raise AssertionError(what)


def raises(call, exc, message=None):
    try:
        call()
    except exc as e:
        if message is not None:
            check(str(e) == message, f"expected {message!r}, got {e!r}")
        return
    raise AssertionError(f"{message or exc.__name__} was not raised")


if sys.platform == "win32":
    for name in ("getpgrp", "getpgid", "ctermid"):
        check(not hasattr(os, name), f"windows grew an os.{name}")
    print("OK")
    raise SystemExit

# ── the process group ─────────────────────────────────────────────────────
pgrp = os.getpgrp()
check(isinstance(pgrp, int) and pgrp > 0, f"getpgrp() answered {pgrp!r}")

# `getpgid(0)` names the caller, which is what `getpgrp()` answers. Two stubs
# both answering None would agree here too, hence the type check above.
check(os.getpgid(0) == pgrp, f"getpgid(0)={os.getpgid(0)!r} but getpgrp()={pgrp!r}")
check(os.getpgid(os.getpid()) == pgrp, "getpgid(getpid()) is not the caller's group")

raises(lambda: os.getpgid(), TypeError)
raises(lambda: os.getpgid("x"), TypeError)
# No process has this id, and the host says so rather than answering a number.
raises(lambda: os.getpgid(0x7FFFFFFE), OSError)

# ── the controlling terminal ──────────────────────────────────────────────
tty = os.ctermid()
check(isinstance(tty, str), f"ctermid() answered {tty!r}")
check(tty.startswith("/"), f"ctermid() is not a path: {tty!r}")
check(os.path.basename(tty) != "", f"ctermid() names no device: {tty!r}")

print("OK")
