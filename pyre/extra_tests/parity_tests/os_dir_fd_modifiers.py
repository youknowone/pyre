"""The name-taking entry points that carry a keyword-only dir_fd.

os.py:124-132 puts mkdir, mkfifo, open, unlink and rmdir in supports_dir_fd
from HAVE_MKDIRAT, HAVE_MKFIFOAT, HAVE_OPENAT and HAVE_UNLINKAT. A modifier
that is accepted and then ignored creates or removes the wrong name, so every
claim is exercised against a directory the working directory does not contain,
and the working directory is checked afterwards for the name that should not
have appeared there.
"""

import atexit
import os
import shutil
import stat
import sys
import tempfile


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


here = os.path.abspath(__file__)
start = os.getcwd()
d = tempfile.mkdtemp()
atexit.register(shutil.rmtree, d, ignore_errors=True)
sub = os.path.join(d, "sub")
os.mkdir(sub)
# The working directory now holds "sub" and nothing else, so any other name
# resolved here cannot be found — which is what makes the assertions below
# distinguish a honoured dir_fd from an ignored one.
os.chdir(d)

if sys.platform == "win32":
    # Windows resolves no name against a descriptor; os_supports_fd.py covers
    # the arm that says so.
    for fn in (os.open, os.mkdir, os.rmdir, os.unlink):
        check(fn not in os.supports_dir_fd, f"{fn.__name__} claims dir_fd on windows")
    os.chdir(start)
    print("OK")
    raise SystemExit

dfd = os.open("sub", os.O_RDONLY)
try:
    # ── a directory made and removed through the descriptor ───────────────
    if os.mkdir in os.supports_dir_fd:
        os.mkdir("m", dir_fd=dfd)
        check(os.path.isdir(os.path.join(sub, "m")), "mkdir(dir_fd) missed the descriptor")
        check(not os.path.exists("m"), "mkdir(dir_fd) resolved against the cwd")
        # Both are the same unlinkat, told apart by AT_REMOVEDIR, so a build
        # cannot claim one and not the other.
        check(os.rmdir in os.supports_dir_fd, "mkdir claims dir_fd but rmdir does not")
        os.rmdir("m", dir_fd=dfd)
        check(not os.path.exists(os.path.join(sub, "m")), "rmdir(dir_fd) removed nothing")
    else:
        raises(lambda: os.mkdir("m", dir_fd=dfd), NotImplementedError)
        raises(lambda: os.rmdir("m", dir_fd=dfd), NotImplementedError)

    # ── a file opened and unlinked through the descriptor ─────────────────
    if os.open in os.supports_dir_fd:
        os.close(os.open("o", os.O_WRONLY | os.O_CREAT, 0o600, dir_fd=dfd))
        check(os.path.isfile(os.path.join(sub, "o")), "open(dir_fd) missed the descriptor")
        check(not os.path.exists("o"), "open(dir_fd) resolved against the cwd")
        check(os.unlink in os.supports_dir_fd, "open claims dir_fd but unlink does not")
        os.unlink("o", dir_fd=dfd)
        check(not os.path.exists(os.path.join(sub, "o")), "unlink(dir_fd) removed nothing")
    else:
        raises(lambda: os.open("o", os.O_RDONLY, dir_fd=dfd), NotImplementedError)
        raises(lambda: os.unlink("o", dir_fd=dfd), NotImplementedError)

    # ── os.remove takes the modifier without being advertised ─────────────
    # os.py:131-132 names unlink and rmdir only, so the set does not carry
    # remove even though it is the same call written out a second time.
    check(os.remove not in os.supports_dir_fd, "os.py names unlink and rmdir, not remove")
    if os.unlink in os.supports_dir_fd:
        os.close(os.open(os.path.join(sub, "r"), os.O_WRONLY | os.O_CREAT, 0o600))
        os.remove("r", dir_fd=dfd)
        check(not os.path.exists(os.path.join(sub, "r")), "remove(dir_fd) removed nothing")

    # ── a fifo made through the descriptor ────────────────────────────────
    if hasattr(os, "mkfifo") and os.mkfifo in os.supports_dir_fd:
        os.mkfifo("p", 0o600, dir_fd=dfd)
        made = os.lstat(os.path.join(sub, "p"))
        check(stat.S_ISFIFO(made.st_mode), "mkfifo(dir_fd) made something else")
        check(not os.path.exists("p"), "mkfifo(dir_fd) resolved against the cwd")
        os.unlink("p", dir_fd=dfd)

    # ── an absolute name ignores dir_fd rather than being refused ─────────
    if os.mkdir in os.supports_dir_fd:
        os.mkdir(os.path.join(d, "abs"), dir_fd=dfd)
        check(os.path.isdir(os.path.join(d, "abs")), "mkdir(absolute, dir_fd) went elsewhere")
        os.rmdir(os.path.join(d, "abs"), dir_fd=dfd)

    # ── None is the default, not a request, and it is what callers spell ──
    os.mkdir("n", dir_fd=None)
    os.rmdir("n", dir_fd=None)
    os.close(os.open("n", os.O_WRONLY | os.O_CREAT, 0o600, dir_fd=None))
    os.unlink("n", dir_fd=None)
    os.close(os.open("n", os.O_WRONLY | os.O_CREAT, 0o600))
    os.remove("n", dir_fd=None)

    # ── the parameters before it keep their own names ─────────────────────
    os.mkdir(path="k", mode=0o755)
    os.rmdir(path="k")
    os.close(os.open(path=here, flags=os.O_RDONLY, mode=0o777))
    if hasattr(os, "mkfifo"):
        os.mkfifo(path="k", mode=0o600)
        os.unlink(path="k")

    # ── the argument list itself ──────────────────────────────────────────
    # A signature whose positional count is fixed says "exactly"; one with a
    # default says "at most".
    raises(
        lambda: os.open(here, 0, 0, 0),
        TypeError,
        "open() takes at most 3 positional arguments (4 given)",
    )
    raises(
        lambda: os.mkdir("x", 0, 0),
        TypeError,
        "mkdir() takes at most 2 positional arguments (3 given)",
    )
    raises(
        lambda: os.rmdir("a", "b"),
        TypeError,
        "rmdir() takes exactly 1 positional argument (2 given)",
    )
    raises(
        lambda: os.unlink("a", "b"),
        TypeError,
        "unlink() takes exactly 1 positional argument (2 given)",
    )
    raises(
        lambda: os.remove("a", "b"),
        TypeError,
        "remove() takes exactly 1 positional argument (2 given)",
    )
    # dir_fd is keyword-only, so a fourth positional is surplus rather than one.
    raises(
        lambda: os.open(here, 0, 0, dfd),
        TypeError,
        "open() takes at most 3 positional arguments (4 given)",
    )

    raises(
        lambda: os.open(here),
        TypeError,
        "open() missing required argument 'flags' (pos 2)",
    )
    for call, name in (
        (lambda: os.mkdir(), "mkdir"),
        (lambda: os.rmdir(), "rmdir"),
        (lambda: os.unlink(), "unlink"),
        (lambda: os.open(), "open"),
    ):
        raises(call, TypeError, f"{name}() missing required argument 'path' (pos 1)")

    raises(
        lambda: os.mkdir("x", path="y"),
        TypeError,
        "argument for mkdir() given by name ('path') and position (1)",
    )

    # An unknown keyword is named, and so is the call that turned it away.
    for call, name in (
        (lambda: os.open(here, 0, nope=1), "open"),
        (lambda: os.mkdir("x", nope=1), "mkdir"),
        (lambda: os.rmdir("x", nope=1), "rmdir"),
        (lambda: os.unlink("x", nope=1), "unlink"),
    ):
        raises(call, TypeError, f"{name}() got an unexpected keyword argument 'nope'")

    # The modifier is converted before the platform is consulted, so a wrongly
    # typed one is a TypeError even where dir_fd could not have been honoured.
    for call in (
        lambda: os.open(here, 0, dir_fd="x"),
        lambda: os.mkdir("x", dir_fd="x"),
        lambda: os.rmdir("x", dir_fd="x"),
        lambda: os.unlink("x", dir_fd="x"),
    ):
        raises(call, TypeError, "argument should be integer or None, not str")

    # The path boundary names the call it belongs to.
    raises(
        lambda: os.unlink(-1),
        TypeError,
        "unlink: path should be string, bytes or os.PathLike, not int",
    )
    raises(
        lambda: os.mkdir(-1),
        TypeError,
        "mkdir: path should be string, bytes or os.PathLike, not int",
    )

    # A name that is not there reports itself.
    raises(lambda: os.unlink("gone"), FileNotFoundError)
    try:
        os.rmdir("gone", dir_fd=dfd)
    except FileNotFoundError as e:
        check(e.filename == "gone", f"rmdir(dir_fd) filename: {e.filename!r}")
    else:
        raise AssertionError("rmdir on a missing name did not raise")
finally:
    os.close(dfd)
    os.chdir(start)

print("OK")
