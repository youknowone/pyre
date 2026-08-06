"""`OSError`'s fourth argument is a Windows error code where the platform has one.

`interp_exceptions.py:551-652 W_OSError._parse_init_args` reads a 2..=5 argument
call as `errno, strerror, filename, winerror, filename2`.  Where Windows error
codes exist, an integer fourth argument decides the errno through
`rwin32.py:285-306 build_winerror_to_errno` — so it also decides which errno
subclass the instance gets — and is kept in its own `winerror` slot, which
`__str__` reports in place of the errno.  Everywhere else the argument is
accepted and unused, and no `winerror` attribute exists at all.
"""

import sys

WIN32 = sys.platform == "win32"

if not WIN32:
    assert not hasattr(OSError, "winerror")
    assert "winerror" not in dir(OSError)

    # The fourth argument is still accepted, and changes nothing.
    e = OSError(22, "msg", "file", 5)
    assert type(e) is OSError, type(e)
    assert e.errno == 22, e.errno
    assert e.args == (22, "msg"), e.args
    assert e.filename == "file", e.filename
    assert e.filename2 is None, e.filename2
    assert str(e) == "[Errno 22] msg: 'file'", str(e)

    # and the fifth is the second filename, exactly as with four arguments.
    e = OSError(22, "msg", "file", 5, "file2")
    assert e.filename2 == "file2", e.filename2
    assert str(e) == "[Errno 22] msg: 'file' -> 'file2'", str(e)

    print("OK")
    raise SystemExit

assert "winerror" in dir(OSError)
assert type(OSError.__dict__["winerror"]).__name__ == "member_descriptor"
# The descriptor names the class it was defined on, the way the rest of
# `OSError`'s members do — that is what `repr` and `inspect` read it from.
assert OSError.__dict__["winerror"].__objclass__ is OSError
assert OSError.__dict__["errno"].__objclass__ is OSError

# `ERROR_ACCESS_DENIED` maps to `EACCES`, so the class follows the Windows code
# and not the errno that was passed.
e = OSError(22, "msg", "file", 5)
assert type(e) is PermissionError, type(e)
assert e.errno == 13, e.errno
assert e.winerror == 5, e.winerror
assert e.args == (13, "msg"), e.args
assert str(e) == "[WinError 5] msg: 'file'", str(e)

# A code the map does not name takes `EINVAL`.  Zero is such a code, and a
# supplied zero is a code that is present rather than one that is missing.
for winerror in (9999, 0):
    e = OSError(2, "msg", "file", winerror)
    assert type(e) is OSError, type(e)
    assert e.errno == 22, e.errno
    assert e.winerror == winerror, e.winerror
    assert str(e) == "[WinError %d] msg: 'file'" % winerror, str(e)

# A fourth argument that is not an integer is kept as it is and leaves the errno
# alone; `__str__` still reports it as the Windows code.
for winerror in (None, "notint"):
    e = OSError(22, "msg", "file", winerror)
    assert type(e) is OSError, type(e)
    assert e.errno == 22, e.errno
    assert e.winerror == winerror, e.winerror
    assert str(e) == "[WinError %s] msg: 'file'" % (winerror,), str(e)

# A non-integer code leaves the errno alone, so the subclass is the one that
# errno names — the code decides the class only when it is one.
e = OSError(1, "msg", "file", "notint")
assert type(e) is PermissionError, type(e)
assert e.errno == 1, e.errno

# The derived errno replaces the first argument whether or not a filename came
# with it; what a filename changes is how many of them `args` keeps.  Without
# one the whole call is kept, first argument rewritten.
e = OSError(2, "msg", None, 5)
assert type(e) is PermissionError, type(e)
assert e.args == (13, "msg", None, 5), e.args
assert str(e) == "[WinError 5] msg", str(e)

# Both filenames are reported alongside the Windows code.
e = OSError(22, "msg", "file", 5, "file2")
assert e.filename2 == "file2", e.filename2
assert str(e) == "[WinError 5] msg: 'file' -> 'file2'", str(e)

# Fewer than four arguments leaves the slot empty, which is the errno spelling.
e = OSError(2, "msg")
assert type(e) is FileNotFoundError, type(e)
assert e.winerror is None, e.winerror
assert str(e) == "[Errno 2] msg", str(e)

# The slot is writable, and writing it does not touch the errno.
e.winerror = 1234
assert e.winerror == 1234, e.winerror
assert e.errno == 2, e.errno
assert str(e) == "[WinError 1234] msg", str(e)

# Emptying it restores the errno spelling rather than leaving a `None` code.
del e.winerror
assert e.winerror is None, e.winerror
assert str(e) == "[Errno 2] msg", str(e)

# The code is reported only where there is something to report it with: a
# filename spells a missing strerror as `None`, a strerror renders on its own,
# and with neither the errno arms answer as though no code had been set.
blank = OSError()
blank.winerror = 5
assert str(blank) == "", str(blank)
blank.errno = 2
assert str(blank) == "", str(blank)
blank.filename = "f"
assert str(blank) == "[WinError 5] None: 'f'", str(blank)
blank.filename2 = "g"
assert str(blank) == "[WinError 5] None: 'f' -> 'g'", str(blank)
blank.strerror = "s"
assert str(blank) == "[WinError 5] s: 'f' -> 'g'", str(blank)

# The rebound errno is what a pickle round-trip carries, and the Windows code is
# not part of it.
assert OSError(22, "msg", "file", 5).__reduce__() == (
    PermissionError,
    (13, "msg", "file"),
)

print("OK")
