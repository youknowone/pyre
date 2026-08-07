"""os.spawnv and the modes it takes come from os.py, not from posix.

os.py:881 writes the spawn family in Python over fork+exec+waitpid, and does so
only `if not _exists("spawnv")` — so a `spawnv` bound by the extension module
does not get overwritten by that definition, it prevents it. The same block
defines P_WAIT and P_NOWAIT, which is why binding those below it is the same
mistake: the two modes have to differ for P_NOWAIT to mean anything.
"""

import os
import sys


def check(cond, what):
    if not cond:
        raise AssertionError(what)


check(hasattr(os, "spawnv"), "no os.spawnv")
check(os.P_WAIT != os.P_NOWAIT, f"P_WAIT and P_NOWAIT are both {os.P_WAIT}")
check(os.__all__.count("P_WAIT") == 1, "P_WAIT is in __all__ twice")
check(os.__all__.count("spawnv") == 1, "spawnv is in __all__ twice")

if sys.platform == "win32" or not hasattr(os, "fork"):
    # Windows spawns through `_spawnv` rather than fork+exec, and takes the C
    # runtime's own `_P_*` numbering, where `P_NOWAITO` is 3 rather than the 1
    # os.py gives it. os_supports_fd.py covers what that build does carry.
    print("OK")
    raise SystemExit

# os.py defines `P_NOWAIT = P_NOWAITO = 1` in the branch that has fork, so the
# two are one value there and a module that bound them apart would say so.
check(os.P_NOWAIT == os.P_NOWAITO, "P_NOWAIT and P_NOWAITO disagree")

exe = sys.executable
check(bool(exe), "no sys.executable to spawn")

# P_WAIT hands back what the child exited with.
code = os.spawnv(os.P_WAIT, exe, [exe, "-c", "raise SystemExit(7)"])
check(code == 7, f"spawnv(P_WAIT) returned {code!r}, not the child's exit code")

code = os.spawnv(os.P_WAIT, exe, [exe, "-c", ""])
check(code == 0, f"spawnv(P_WAIT) on a clean exit returned {code!r}")

# P_NOWAIT hands back a pid the caller waits on itself. A mode that was equal
# to P_WAIT would return the exit code here instead, and the wait would fail.
pid = os.spawnv(os.P_NOWAIT, exe, [exe, "-c", "raise SystemExit(3)"])
check(isinstance(pid, int) and pid > 0, f"spawnv(P_NOWAIT) returned {pid!r}, not a pid")
waited, status = os.waitpid(pid, 0)
check(waited == pid, f"waitpid returned {waited!r} for {pid!r}")
check(os.waitstatus_to_exitcode(status) == 3, f"child status {status!r}")

# spawnve carries an environment of its own.
code = os.spawnve(
    os.P_WAIT,
    exe,
    [exe, "-c", "import os,sys; sys.exit(int(os.environ['PARITY_MARK']))"],
    {**os.environ, "PARITY_MARK": "5"},
)
check(code == 5, f"spawnve did not pass the environment: {code!r}")

# spawnvp looks the name up along $PATH, so a bare name resolves.
if hasattr(os, "spawnvp"):
    code = os.spawnvp(os.P_WAIT, "sh", ["sh", "-c", "exit 4"])
    check(code == 4, f"spawnvp did not find sh along PATH: {code!r}")

# The argument list is checked before the fork.
for bad, exc in ((("not a list"), TypeError), (([]), ValueError), (([""]), ValueError)):
    try:
        os.spawnv(os.P_WAIT, exe, bad)
    except exc:
        pass
    else:
        raise AssertionError(f"spawnv accepted {bad!r}")

print("OK")
