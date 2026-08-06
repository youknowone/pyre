"""os.chflags, os.lchflags and os.mknod either work or are not there.

A stub that takes any argument and reports success is worse than an absent
name, because the callers probe for presence and believe the answer:
`shutil.copystat` (shutil.py:467) and `tempfile._resetperms`
(tempfile.py:276-282) both reach chflags that way, and `tarfile` reaches mknod
through `hasattr(os, "mknod")`. So each name that exists is exercised here, and
each one that does not is checked for being absent on both sides of its pair.

os.py:126 puts mknod in supports_dir_fd from HAVE_MKNODAT; os.py:182 puts
chflags in supports_follow_symlinks from HAVE_LCHFLAGS.
"""

import os
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


start = os.getcwd()
d = tempfile.mkdtemp()
p = os.path.join(d, "f")
with open(p, "wb") as f:
    f.write(b"x")

if sys.platform == "win32":
    for name in ("chflags", "lchflags", "mknod"):
        check(not hasattr(os, name), f"windows grew an os.{name}")
    print("OK")
    raise SystemExit

os.chdir(d)
link = os.path.join(d, "l")
os.symlink("f", link)
sub = os.path.join(d, "sub")
os.mkdir(sub)

# A name that is present converts its path, so a float is turned away rather
# than accepted and ignored.
for name in ("chflags", "lchflags", "mknod"):
    fn = getattr(os, name, None)
    if fn is None:
        continue
    raises(
        lambda fn=fn, name=name: fn(1.5, 0),
        TypeError,
        f"{name}: path should be string, bytes or os.PathLike, not float",
    )

# ── mknod makes a node ────────────────────────────────────────────────────
if hasattr(os, "mknod"):
    # Only a FIFO is unprivileged; a plain mode asks for a regular file and
    # the kernel refuses that to anyone but root, which is why the default
    # form is not exercised.
    os.mknod("n", 0o600 | stat.S_IFIFO)
    check(stat.S_ISFIFO(os.lstat(os.path.join(d, "n")).st_mode), "mknod made no fifo")
    os.unlink("n")

    os.mknod(path="n", mode=0o600 | stat.S_IFIFO, device=0)
    check(stat.S_ISFIFO(os.lstat(os.path.join(d, "n")).st_mode), "mknod by keyword made no fifo")
    os.unlink("n")

    if os.mknod in os.supports_dir_fd:
        dfd = os.open("sub", os.O_RDONLY)
        try:
            os.mknod("n", 0o600 | stat.S_IFIFO, 0, dir_fd=dfd)
            check(os.path.exists(os.path.join(sub, "n")), "mknod(dir_fd) missed the descriptor")
            check(not os.path.exists(os.path.join(d, "n")), "mknod(dir_fd) resolved against the cwd")
            os.unlink("n", dir_fd=dfd)
        finally:
            os.close(dfd)

    raises(
        lambda: os.mknod("x", 0, 0, 0),
        TypeError,
        "mknod() takes at most 3 positional arguments (4 given)",
    )
    raises(
        lambda: os.mknod(),
        TypeError,
        "mknod() missing required argument 'path' (pos 1)",
    )
    raises(
        lambda: os.mknod("x", nope=1),
        TypeError,
        "mknod() got an unexpected keyword argument 'nope'",
    )
    raises(lambda: os.mknod("x", dir_fd="x"), TypeError, "argument should be integer or None, not str")

# ── chflags sets the flag it is given ─────────────────────────────────────
if hasattr(os, "chflags"):
    check(hasattr(os, "lchflags"), "chflags without lchflags")
    check(hasattr(os.stat(p), "st_flags"), "chflags without st_flags to read it back")

    os.chflags(p, stat.UF_NODUMP)
    check(os.stat(p).st_flags & stat.UF_NODUMP, "chflags set nothing")
    os.chflags(p, 0)
    check(not os.stat(p).st_flags & stat.UF_NODUMP, "chflags cleared nothing")

    # follow_symlinks is a positional-or-keyword parameter here, not a
    # keyword-only one, and it reaches the link rather than its target.
    if os.chflags in os.supports_follow_symlinks:
        os.chflags(link, stat.UF_NODUMP, follow_symlinks=False)
        check(os.lstat(link).st_flags & stat.UF_NODUMP, "chflags(follow=False) missed the link")
        check(not os.stat(p).st_flags & stat.UF_NODUMP, "chflags(follow=False) changed the target")
        os.lchflags(link, 0)
        check(not os.lstat(link).st_flags & stat.UF_NODUMP, "lchflags cleared nothing")
        os.chflags(link, 0, False)
    else:
        check(not hasattr(os, "lchflags"), "lchflags without supports_follow_symlinks")

    os.chflags(path=p, flags=0)
    os.chflags(p, 0, True)

    # Neither takes a keyword-only argument, so every argument counts against
    # the one limit — an extra keyword is over it rather than unknown.
    raises(
        lambda: os.lchflags(link, 0, follow_symlinks=False),
        TypeError,
        "lchflags() takes at most 2 arguments (3 given)",
    )
    raises(
        lambda: os.chflags(p, 0, 0, 0),
        TypeError,
        "chflags() takes at most 3 arguments (4 given)",
    )
    raises(
        lambda: os.chflags(p),
        TypeError,
        "chflags() missing required argument 'flags' (pos 2)",
    )
    raises(
        lambda: os.chflags(p, 0, nope=1),
        TypeError,
        "chflags() got an unexpected keyword argument 'nope'",
    )

    # A missing name reports itself.
    try:
        os.chflags(os.path.join(d, "nope"), 0)
    except FileNotFoundError as e:
        check(e.filename == os.path.join(d, "nope"), f"chflags filename: {e.filename!r}")
    else:
        raise AssertionError("chflags on a missing name did not raise")
else:
    check(not hasattr(os, "lchflags"), "lchflags without chflags")

os.chdir(start)
print("OK")
