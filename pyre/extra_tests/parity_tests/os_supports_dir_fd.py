"""Every name in os.supports_dir_fd / os.supports_follow_symlinks must honour it.

os.py:116-134 builds supports_dir_fd and os.py:157-193 supports_follow_symlinks
out of posix._have_functions, and shutil.copystat (shutil.py:409-446) reads the
second one to decide whether it can copy a symlink's own metadata or has to
silently skip that step. A name in either set whose entry point raises
NotImplementedError sends the caller down a route that cannot work; a name
missing from the set costs a caller that fallback for no reason.

The modifiers are checked by observation rather than by return value: a dir_fd
call is repeated against a name that does not exist under the descriptor, which
is what separates "resolved the name against dir_fd" from "ignored it and hit
the same file through the cwd", and follow_symlinks=False is read back through
lstat against a target whose times must not have moved.
"""

import os
import shutil
import sys
import tempfile


def check(cond, what):
    if not cond:
        raise AssertionError(what)


def raises(exc, call, what):
    try:
        call()
    except exc as e:
        return e
    except BaseException as e:  # noqa: BLE001 - the type is the assertion
        raise AssertionError(f"{what}: raised {type(e).__name__}: {e}") from None
    raise AssertionError(f"{what}: did not raise")


# The whole *at family is POSIX; on Windows os.py seeds these sets from
# MS_WINDOWS and HAVE_LSTAT alone, and there is no dir_fd form to exercise.
if sys.platform == "win32":
    check(os.stat in os.supports_follow_symlinks, "stat missing from supports_follow_symlinks")
    check(not os.supports_dir_fd, "supports_dir_fd is not empty on windows")
    print("OK")
    raise SystemExit

d = tempfile.mkdtemp()
p = os.path.join(d, "f")
with open(p, "wb") as f:
    f.write(b"0123456789")
link = os.path.join(d, "l")
os.symlink("f", link)

# stat is HAVE_FSTATAT, chown is HAVE_FCHOWNAT, utime is HAVE_UTIMENSAT — the
# three *at calls that resolve a name against a directory descriptor. Each of
# the three macros feeds both sets, so the floor is the same on both sides.
for name in ("chown", "stat", "utime"):
    fn = getattr(os, name)
    check(fn in os.supports_dir_fd, f"{name} missing from supports_dir_fd")
    check(fn in os.supports_follow_symlinks, f"{name} missing from supports_follow_symlinks")

# Whole seconds, so a filesystem that stores no sub-second component still
# reads back the value that was written.
ATIME, MTIME = 1_000_000_000, 2_000_000_000
LINK_ATIME, LINK_MTIME = 3_000_000_000, 4_000_000_000

dfd = os.open(d, os.O_RDONLY)
try:
    os.utime("f", ns=(ATIME, MTIME), dir_fd=dfd)
    st = os.stat(p)
    check(st.st_atime_ns == ATIME, f"utime(dir_fd) atime {st.st_atime_ns}")
    check(st.st_mtime_ns == MTIME, f"utime(dir_fd) mtime {st.st_mtime_ns}")
    raises(
        FileNotFoundError,
        lambda: os.utime("absent", ns=(ATIME, MTIME), dir_fd=dfd),
        "utime resolved a name that does not exist under dir_fd",
    )

    # uid/gid of -1 leaves both unchanged, so this needs no privilege and its
    # only observable is whether the call reaches the file at all.
    os.chown("f", -1, -1, dir_fd=dfd)
    raises(
        FileNotFoundError,
        lambda: os.chown("absent", -1, -1, dir_fd=dfd),
        "chown resolved a name that does not exist under dir_fd",
    )
    os.stat("f", dir_fd=dfd)
    raises(
        FileNotFoundError,
        lambda: os.stat("absent", dir_fd=dfd),
        "stat resolved a name that does not exist under dir_fd",
    )

    # follow_symlinks=False rewrites the link itself; the target keeps the
    # times the dir_fd calls above gave it.
    os.utime(link, ns=(LINK_ATIME, LINK_MTIME), follow_symlinks=False)
    check(os.stat(p).st_mtime_ns == MTIME, "utime(follow_symlinks=False) moved the target")
    lst = os.lstat(link)
    check(lst.st_mtime_ns == LINK_MTIME, f"utime(follow_symlinks=False) mtime {lst.st_mtime_ns}")
    os.chown(link, -1, -1, follow_symlinks=False)
    check(os.stat(link, follow_symlinks=False).st_mtime_ns == LINK_MTIME, "lstat via stat()")

    # The two modifiers compose: a relative name under dir_fd, not followed.
    os.utime("l", ns=(LINK_ATIME, LINK_MTIME + 1), dir_fd=dfd, follow_symlinks=False)
    check(os.lstat(link).st_mtime_ns == LINK_MTIME + 1, "utime(dir_fd, follow_symlinks=False)")
    check(os.stat(p).st_mtime_ns == MTIME, "utime(dir_fd, follow_symlinks=False) moved the target")
    os.chown("l", -1, -1, dir_fd=dfd, follow_symlinks=False)

    # A descriptor already names the file, so neither modifier — each of which
    # reinterprets a *name* — can apply.
    fd = os.open(p, os.O_RDWR)
    try:
        e = raises(
            ValueError,
            lambda: os.utime(fd, ns=(ATIME, MTIME), dir_fd=dfd),
            "utime(fd, dir_fd=...)",
        )
        check(
            str(e) == "utime: can't specify dir_fd without matching path",
            f"utime(fd, dir_fd=...) message {str(e)!r}",
        )
        e = raises(
            ValueError,
            lambda: os.chown(fd, -1, -1, dir_fd=dfd),
            "chown(fd, dir_fd=...)",
        )
        check(
            str(e) == "chown: can't specify both dir_fd and fd",
            f"chown(fd, dir_fd=...) message {str(e)!r}",
        )
        e = raises(
            ValueError,
            lambda: os.chown(fd, -1, -1, follow_symlinks=False),
            "chown(fd, follow_symlinks=False)",
        )
        check(
            str(e) == "chown: cannot use fd and follow_symlinks together",
            f"chown(fd, follow_symlinks=False) message {str(e)!r}",
        )
    finally:
        os.close(fd)
finally:
    os.close(dfd)

# What the second set actually buys a caller: with utime in
# supports_follow_symlinks, copystat carries the link's own times across
# instead of quietly doing nothing (shutil.py:435-439 `lookup` returns `_nop`
# for any name the set does not carry, and :446 is the call it stands in for).
other = os.path.join(d, "l2")
os.symlink("f", other)
shutil.copystat(link, other, follow_symlinks=False)
check(
    os.lstat(other).st_mtime_ns == os.lstat(link).st_mtime_ns,
    "copystat(follow_symlinks=False) did not carry the link's own times",
)

print("OK")
