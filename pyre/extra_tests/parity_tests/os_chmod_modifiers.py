"""os.chmod honours dir_fd and follow_symlinks, and says so.

os.py:118 puts chmod in supports_dir_fd from HAVE_FCHMODAT, and os.py:183 puts
it in supports_follow_symlinks from HAVE_LCHMOD — a narrower bit, because a host
can have fchmodat and still not honour AT_SYMLINK_NOFOLLOW (os.py:159-177). Each
claim is exercised here rather than asserted, since a modifier that is accepted
and ignored changes the wrong file.
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


def mode_of(path, follow=True):
    st = os.stat(path) if follow else os.lstat(path)
    return st.st_mode & 0o777


d = tempfile.mkdtemp()
atexit.register(shutil.rmtree, d, ignore_errors=True)
p = os.path.join(d, "f")
with open(p, "wb") as f:
    f.write(b"x")

if sys.platform == "win32":
    # Windows reaches chmod's follow_symlinks through MS_WINDOWS and never
    # honours dir_fd; os_supports_fd.py covers that arm.
    check(os.chmod not in os.supports_dir_fd, "chmod claims dir_fd on windows")
    print("OK")
    raise SystemExit

# ── dir_fd resolves the name against the descriptor ──────────────────────
dfd = os.open(d, os.O_RDONLY)
try:
    if os.chmod in os.supports_dir_fd:
        # "f" does not exist relative to the working directory, so a chmod that
        # ignored dir_fd could not have found it.
        check(not os.path.exists("f"), "the test's premise: no 'f' in the cwd")
        os.chmod("f", 0o600, dir_fd=dfd)
        check(mode_of(p) == 0o600, "chmod(dir_fd) did not reach the file")
        os.chmod("f", 0o644, dir_fd=dfd)
        check(mode_of(p) == 0o644, "chmod(dir_fd) did not reach the file again")

        # An absolute path ignores dir_fd rather than being refused.
        os.chmod(p, 0o600, dir_fd=dfd)
        check(mode_of(p) == 0o600, "chmod(absolute, dir_fd) did not reach the file")
        os.chmod(p, 0o644)

    # None is the default, not a request, and it is what callers spell.
    os.chmod(p, 0o644, dir_fd=None, follow_symlinks=True)

    # ── follow_symlinks reaches the link itself ──────────────────────────
    link = os.path.join(d, "l")
    os.symlink("f", link)
    target_before = mode_of(p)
    if os.chmod in os.supports_follow_symlinks:
        os.chmod(link, 0o600, follow_symlinks=False)
        check(mode_of(link, follow=False) == 0o600, "chmod(follow_symlinks=False) missed the link")
        check(mode_of(p) == target_before, "chmod(follow_symlinks=False) changed the target")

        # os.lchmod is the same call under its own name, and os.py:159 ties the
        # capability bit to that function existing at all.
        check(hasattr(os, "lchmod"), "supports_follow_symlinks without lchmod")
        os.lchmod(link, 0o604)
        check(mode_of(link, follow=False) == 0o604, "lchmod missed the link")
        check(mode_of(p) == target_before, "lchmod changed the target")
        # It takes no keyword of its own.
        try:
            os.lchmod(link, 0o604, follow_symlinks=False)
        except TypeError:
            pass
        else:
            raise AssertionError("lchmod accepted follow_symlinks")
    else:
        check(not hasattr(os, "lchmod"), "lchmod without supports_follow_symlinks")

    # Following the link is the default, and it lands on the target.
    os.chmod(link, 0o640)
    check(mode_of(p) == 0o640, "chmod(link) did not follow to the target")
    os.chmod(p, 0o644)

    # ── a descriptor answers before either modifier ──────────────────────
    # `fchmod` is what os.chmod(fd, …) means, so dir_fd and follow_symlinks are
    # not consulted rather than refused (interp_posix.py:1233-1242) — unlike
    # os.chown, which turns both away.
    if os.chmod in os.supports_fd:
        fd = os.open(p, os.O_RDWR)
        try:
            os.chmod(fd, 0o600, dir_fd=dfd)
            check(mode_of(p) == 0o600, "chmod(fd, dir_fd) did not use the descriptor")
            os.chmod(fd, 0o640, follow_symlinks=False)
            check(mode_of(p) == 0o640, "chmod(fd, follow_symlinks) did not use the descriptor")
        finally:
            os.close(fd)
        os.chmod(p, 0o644)

    # ── the argument list itself ─────────────────────────────────────────
    try:
        os.chmod(p, 0o644, nope=1)
    except TypeError as e:
        check(
            str(e) == "chmod() got an unexpected keyword argument 'nope'",
            f"chmod unknown keyword: {e}",
        )
    else:
        raise AssertionError("chmod accepted an unknown keyword")

    # Both modifiers are keyword-only.
    try:
        os.chmod(p, 0o644, None)
    except TypeError:
        pass
    else:
        raise AssertionError("chmod took dir_fd positionally")

    # A bad dir_fd is converted before the platform is reported.
    try:
        os.chmod("f", 0o644, dir_fd="x")
    except TypeError:
        pass
    else:
        raise AssertionError("chmod accepted a str dir_fd")

    # The mode keeps its own name.
    os.chmod(path=p, mode=0o644)
    check(mode_of(p) == 0o644, "chmod by keyword did not reach the file")
finally:
    os.close(dfd)

# A missing name reports itself.
try:
    os.chmod(os.path.join(d, "nope"), 0o644)
except FileNotFoundError as e:
    check(e.filename == os.path.join(d, "nope"), f"chmod filename: {e.filename!r}")
else:
    raise AssertionError("chmod on a missing name did not raise")

# `stat` is imported for the mode bits the win32 arm of os_supports_fd.py reads;
# here it keeps the symlink assertion honest about what it compared.
check(stat.S_ISLNK(os.lstat(os.path.join(d, "l")).st_mode), "the link stopped being one")

print("OK")
