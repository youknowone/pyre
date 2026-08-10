"""Nine posix entry points bind their arguments, and say so the same way.

Each of these took its arguments straight off the raw slice: the trailing
`__pyre_kw__` marker dict was never split away, so a keyword either vanished or
arrived as a positional value, and a surplus positional was dropped instead of
refused. The loudest case was `symlink(src, dst, target_is_directory, dir_fd)`,
which created the link and returned None where CPython raises.

Unlike the dup2 test, this one *does* assert message text. These entry points
bind through the posix module's own binder, which carries the clinic spellings,
so text parity is reachable here — and the two spellings are not
interchangeable: an entry point with a keyword-bindable parameter reports
`f() takes at most 1 argument (2 given)`, one that is entirely positional-only
reports `f expected at most 1 argument, got 2`. Asserting only the type would
let the two swap places unnoticed.
"""

import os
import sys

ERRORS = []


def check(cond, what):
    if not cond:
        ERRORS.append(what)


def raises(what, expected, fn):
    """Assert fn() raises TypeError whose message is exactly `expected`."""
    try:
        fn()
    except TypeError as e:
        check(str(e) == expected, f"{what}: got {str(e)!r}, expected {expected!r}")
        return
    except Exception as e:
        ERRORS.append(f"{what}: raised {type(e).__name__}({e}), expected TypeError")
        return
    ERRORS.append(f"{what}: no exception, expected TypeError")


d = os.getcwd()

# ── the keyword-bindable family: `f() takes at most N argument(s) (M given)` ──
raises(
    "listdir surplus",
    "listdir() takes at most 1 argument (2 given)",
    lambda: os.listdir(d, 1),
)
raises(
    "listdir unknown keyword",
    "listdir() got an unexpected keyword argument 'zzz'",
    lambda: os.listdir(zzz=1),
)
check(isinstance(os.listdir(path=d), list), "listdir(path=) did not bind")

raises(
    "scandir surplus",
    "scandir() takes at most 1 argument (2 given)",
    lambda: os.scandir(d, 1),
)
with os.scandir(path=d) as it:
    check(any(True for _ in it), "scandir(path=) did not bind")

# ── a keyword-only tail makes the count positional-only ──────────────────
raises(
    "access third positional",
    "access() takes exactly 2 positional arguments (3 given)",
    lambda: os.access(d, os.F_OK, True),
)
raises(
    "access unknown keyword",
    "access() got an unexpected keyword argument 'zzz'",
    lambda: os.access(d, os.F_OK, zzz=1),
)
check(os.access(path=d, mode=os.F_OK) is True, "access(path=, mode=) did not bind")

raises(
    "readlink second positional",
    "readlink() takes exactly 1 positional argument (2 given)",
    lambda: os.readlink(d, 1),
)

raises(
    "symlink fourth positional",
    "symlink() takes at most 3 positional arguments (4 given)",
    lambda: os.symlink("a", "b", False, 1),
)

raises(
    "sendfile missing count",
    "sendfile() missing required argument 'count' (pos 4)",
    lambda: os.sendfile(1, 2, 3),
)

# ── positional-only: no `()`, no parenthesised count ─────────────────────
raises(
    "get_terminal_size surplus",
    "get_terminal_size expected at most 1 argument, got 2",
    lambda: os.get_terminal_size(1, 2),
)
raises(
    "get_terminal_size by keyword",
    "posix.get_terminal_size() takes no keyword arguments",
    lambda: os.get_terminal_size(fd=1),
)

# ── positional-only prefix + keyword-only tail ───────────────────────────
for name in ("posix_spawn", "posix_spawnp"):
    spawn = getattr(os, name, None)
    if spawn is None:
        continue
    raises(
        f"{name} too few",
        f"{name}() takes exactly 3 positional arguments (2 given)",
        lambda spawn=spawn: spawn("/bin/true", ["x"]),
    )
    raises(
        f"{name} names are positional-only",
        f"{name}() takes exactly 3 positional arguments (0 given)",
        lambda spawn=spawn: spawn(path="/bin/true", argv=["x"], env={}),
    )
    raises(
        f"{name} unknown keyword",
        f"{name}() got an unexpected keyword argument 'zzz'",
        lambda spawn=spawn: spawn("/bin/true", ["true"], {}, zzz=1),
    )

# ── symlink actually refuses, rather than creating the link ──────────────
# The surplus-positional assertion above passes just as well if the call
# raises *after* doing the work, so the effect is checked separately.
target = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"pyre_sym_{os.getpid()}")
try:
    os.unlink(target)
except OSError:
    pass
try:
    os.symlink("nowhere", target, False, 1)
except TypeError:
    pass
check(not os.path.islink(target), "symlink raised but still created the link")
# ...and that the by-name form does create it, so the check above is not
# passing because symlink stopped working.
os.symlink("nowhere", dst=target)
check(os.path.islink(target), "symlink(dst=) did not create the link")
os.unlink(target)

if ERRORS:
    for e in ERRORS:
        print("FAIL:", e, file=sys.stderr)
    raise AssertionError(f"{len(ERRORS)} binding divergence(s)")

print("OK")
