"""Every name in os.supports_fd must actually accept an open descriptor.

os.py:140-155 builds the set from posix._have_functions, and callers read it to
choose an fd-relative implementation over a path-based one. A name in the set
whose entry point rejects an integer — or, worse, accepts one and does nothing —
sends the caller down a route that cannot work.
"""

import os
import stat
import sys
import tempfile


def check(cond, what):
    if not cond:
        raise AssertionError(what)


# What each entry point wants after its path, for the calls that probe the path
# argument alone.
REST_ARGS = {
    "chdir": (),
    "chmod": (0o644,),
    "chown": (-1, -1),
    "pathconf": ("PC_NAME_MAX",),
    "stat": (),
    "statvfs": (),
    "truncate": (0,),
    "utime": (),
}

# The two whose path argument also takes None, which widens the same list again.
NULLABLE_ARGS = {
    "listdir": (),
    "scandir": (),
}

ALL_ARGS = {**REST_ARGS, **NULLABLE_ARGS}

d = tempfile.mkdtemp()
p = os.path.join(d, "f")
with open(p, "wb") as f:
    f.write(b"0123456789")

# Windows reaches supports_fd through three of os.py's rules rather than the
# HAVE_F* family: `stat` unconditionally (os.py:148, "fstat always works"),
# `chmod` through MS_WINDOWS (os.py:143), and `truncate` through HAVE_FTRUNCATE
# (os.py:149). The descriptor there is a CRT one wrapping a handle rather than a
# kernel fd, which is exactly why each of the three needs exercising and not
# just asserting.
if sys.platform == "win32":
    WIN32_PROBES = {"stat", "chmod", "truncate"}
    names = {f.__name__ for f in os.supports_fd}
    # A name advertised here and not exercised below would be a capability
    # claimed and never tested — the defect this whole file exists to catch.
    check(
        names <= WIN32_PROBES,
        f"supports_fd names with no probe here: {sorted(names - WIN32_PROBES)}",
    )
    check(WIN32_PROBES <= names, f"missing from supports_fd: {sorted(WIN32_PROBES - names)}")

    fd = os.open(p, os.O_RDWR)
    try:
        check(os.stat(fd).st_size == 10, "stat(fd) size")
        check(os.fstat(fd).st_size == 10, "fstat(fd) size")

        # MS_WINDOWS makes the same claim twice — the descriptor form
        # (os.py:143) and follow_symlinks (os.py:184) — so both are read back.
        check(os.chmod in os.supports_follow_symlinks, "chmod missing from supports_follow_symlinks")
        # The mode Windows keeps is the owner's write bit, which is the
        # read-only attribute inverted.
        os.chmod(fd, 0o444)
        check(not os.stat(p).st_mode & stat.S_IWRITE, "chmod(fd) did not clear the write bit")
        os.chmod(fd, 0o644)
        check(os.stat(p).st_mode & stat.S_IWRITE, "chmod(fd) did not restore the write bit")

        # follow_symlinks reaches the name's own attributes. A plain file is
        # its own final component, so this needs no symlink privilege and the
        # bit still has to move.
        os.chmod(p, 0o444, follow_symlinks=False)
        check(not os.stat(p).st_mode & stat.S_IWRITE, "chmod(follow_symlinks=False) did not clear it")
        os.chmod(p, 0o644, follow_symlinks=False)
        check(os.stat(p).st_mode & stat.S_IWRITE, "chmod(follow_symlinks=False) did not restore it")

        # dir_fd is the one modifier Windows cannot honour, and os.py never
        # puts chmod in supports_dir_fd there.
        check(os.chmod not in os.supports_dir_fd, "chmod claims dir_fd on windows")
        try:
            os.chmod(p, 0o644, dir_fd=fd)
        except NotImplementedError:
            pass
        else:
            raise AssertionError("chmod accepted dir_fd on windows")

        os.truncate(fd, 4)
        check(os.stat(fd).st_size == 4, "truncate(fd) did not shrink the file")
        os.truncate(p, 2)
        check(os.stat(p).st_size == 2, "truncate(path) did not shrink the file")

        # The widened allowed-type list appears wherever the descriptor form
        # does, and names the entry point that answered.
        for name in sorted(WIN32_PROBES):
            try:
                getattr(os, name)(1.5, *REST_ARGS[name])
            except TypeError as e:
                check(
                    str(e)
                    == f"{name}: path should be string, bytes, os.PathLike or integer, not float",
                    f"{name} type error: {e}",
                )
            else:
                raise AssertionError(f"{name}(1.5) did not raise")
    finally:
        os.close(fd)
    print("OK")
    raise SystemExit

names = {f.__name__ for f in os.supports_fd}
# `stat` is unconditional (os.py:148 "fstat always works"); the rest are the
# HAVE_F* bits, and every POSIX host defines all of them. Asserting the floor
# up front is what keeps the guarded blocks below from covering less and less
# in silence if a capability is ever dropped rather than fixed.
check(
    set(ALL_ARGS) <= names,
    f"missing from supports_fd: {sorted(set(ALL_ARGS) - names)}",
)

# Two more names so a descriptor's entries have each kind to report.
os.mkdir(os.path.join(d, "sub"))
os.symlink("f", os.path.join(d, "lnk"))
CONTENTS = sorted(["f", "sub", "lnk"])

fd = os.open(p, os.O_RDWR)
dfd = os.open(d, os.O_RDONLY)
try:
    # ── the descriptor form does the work ────────────────────────────────
    if "stat" in names:
        check(os.stat(fd).st_size == 10, "stat(fd) size")

    if "chdir" in names:
        before = os.getcwd()
        os.chdir(dfd)
        check(os.path.samefile(os.getcwd(), d), "chdir(dfd) did not move")
        os.chdir(before)

    if "chmod" in names:
        os.chmod(fd, 0o600)
        check(os.stat(fd).st_mode & 0o777 == 0o600, "chmod(fd) mode")
        os.chmod(fd, 0o644)
        check(os.stat(fd).st_mode & 0o777 == 0o644, "chmod(fd) mode again")

    if "chown" in names:
        st = os.stat(fd)
        # -1/-1 is the "leave unchanged" pair, so this needs no privilege.
        os.chown(fd, -1, -1)
        check(os.stat(fd).st_uid == st.st_uid, "chown(fd) changed the owner")

    if "pathconf" in names:
        check(os.pathconf(fd, "PC_NAME_MAX") > 0, "pathconf(fd)")

    if "statvfs" in names:
        check(os.statvfs(fd).f_bsize > 0, "statvfs(fd)")

    if "truncate" in names:
        os.truncate(fd, 4)
        check(os.stat(fd).st_size == 4, "truncate(fd) did not shrink the file")
        os.truncate(p, 2)
        check(os.stat(p).st_size == 2, "truncate(path) did not shrink the file")

    if "utime" in names:
        os.utime(fd, (12345, 54321))
        check(os.stat(fd).st_mtime == 54321, "utime(fd) mtime")
        check(os.stat(fd).st_atime == 12345, "utime(fd) atime")

    if "listdir" in names:
        check(sorted(os.listdir(dfd)) == CONTENTS, "listdir(dfd) did not read the directory")
        # fdopendir takes the descriptor over and closedir closes it, so the
        # call works on a duplicate — which shares the directory offset, and
        # would leave the caller's own at the end. A second read is what says
        # the offset was put back (rposix.py:844 `_listdir(rewind=True)`).
        check(sorted(os.listdir(dfd)) == CONTENTS, "listdir(dfd) is not repeatable")
        # A descriptor carries no encoding to answer in, so the names are str.
        check(all(isinstance(n, str) for n in os.listdir(dfd)), "listdir(dfd) names are not str")

    if "scandir" in names:
        with os.scandir(dfd) as it:
            entries = {e.name: e for e in it}
        check(sorted(entries) == CONTENTS, "scandir(dfd) did not read the directory")
        # interp_scandir.py:50 leaves the path prefix empty for a descriptor —
        # there is no directory name to join onto.
        check(all(e.path == e.name for e in entries.values()), "scandir(dfd) entry.path")

        # Each entry resolves its own name against that descriptor rather than
        # the working directory, which is the only reason these can answer at
        # all: none of these names exists relative to the cwd.
        check(not any(os.path.exists(n) for n in CONTENTS), "cwd already holds the probe names")
        e = entries["f"]
        check(e.is_file(), "scandir(dfd) file is_file")
        check(not e.is_dir(), "scandir(dfd) file is_dir")
        check(e.stat().st_size == os.stat(p).st_size, "scandir(dfd) file stat")
        check(e.inode() == os.stat(p).st_ino, "scandir(dfd) file inode")

        check(entries["sub"].is_dir(), "scandir(dfd) subdir is_dir")
        check(not entries["sub"].is_file(), "scandir(dfd) subdir is_file")

        lnk = entries["lnk"]
        check(lnk.is_symlink(), "scandir(dfd) symlink is_symlink")
        # follow_symlinks defaults to True, so the target's kind is what shows.
        check(lnk.is_file(), "scandir(dfd) symlink is_file")
        check(not lnk.is_file(follow_symlinks=False), "scandir(dfd) symlink is_file(nofollow)")
        check(
            stat.S_ISLNK(lnk.stat(follow_symlinks=False).st_mode),
            "scandir(dfd) symlink lstat mode",
        )

    # ── the same two also take None, which is what nullable means ────────
    before = os.getcwd()
    os.chdir(d)
    try:
        if "listdir" in names:
            check(sorted(os.listdir(None)) == CONTENTS, "listdir(None) did not read the cwd")
        if "scandir" in names:
            with os.scandir(None) as it:
                check(sorted(e.name for e in it) == CONTENTS, "scandir(None) did not read the cwd")
    finally:
        os.chdir(before)

    # The failure a stat raises names the entry (interp_scandir.py:328
    # `wrap_oserror2(…, self.fget_path(space))`) — which for a descriptor is
    # the bare name, since that is the whole of its path. `is_dir`/`is_file`
    # are not read here: those answer from the type readdir already reported,
    # so a removal does not reach them.
    if "scandir" in names:
        gone = os.path.join(d, "gone")
        open(gone, "w").close()
        with os.scandir(dfd) as it:
            entry = next(e for e in it if e.name == "gone")
        os.remove(gone)
        try:
            entry.stat()
        except FileNotFoundError as e:
            check(e.filename == "gone", f"vanished entry filename: {e.filename!r}")
        else:
            raise AssertionError("stat of a vanished entry did not raise")

    # A descriptor that is not a directory is refused, and since the descriptor
    # is what named the directory, it is what names the failure.
    for name in sorted(names & set(NULLABLE_ARGS)):
        try:
            getattr(os, name)(fd)
        except NotADirectoryError as e:
            check(e.filename == fd, f"{name}(non-directory fd) filename: {e.filename!r}")
        else:
            raise AssertionError(f"{name}(non-directory fd) did not raise")

    # ── an argument that is neither a path nor a descriptor ──────────────
    # path_or_fd names its caller and widens the allowed-type list with the
    # descriptor form, so the message itself reports the capability. Only the
    # names carrying a recipe below are exercised — execve takes an argv.
    for name in sorted(names & set(REST_ARGS)):
        fn = getattr(os, name)
        try:
            fn(1.5, *REST_ARGS[name])
        except TypeError as e:
            check(
                str(e) == f"{name}: path should be string, bytes, os.PathLike or integer, not float",
                f"{name} type error: {e}",
            )
        else:
            raise AssertionError(f"{name}(1.5) did not raise")

    # A nullable path names None in the same list (interp_posix.py:170-172).
    for name in sorted(names & set(NULLABLE_ARGS)):
        fn = getattr(os, name)
        try:
            fn(1.5, *NULLABLE_ARGS[name])
        except TypeError as e:
            check(
                str(e)
                == f"{name}: path should be string, bytes, os.PathLike, integer or None, not float",
                f"{name} type error: {e}",
            )
        else:
            raise AssertionError(f"{name}(1.5) did not raise")

    # A path-only entry point names itself too, without "or integer".
    try:
        os.lchown(1.5, -1, -1)
    except TypeError as e:
        check(
            str(e) == "lchown: path should be string, bytes or os.PathLike, not float",
            f"lchown type error: {e}",
        )
    else:
        raise AssertionError("lchown(1.5) did not raise")

    # ── a descriptor cannot carry the two path-relative modifiers ────────
    if "chown" in names:
        try:
            os.chown(fd, -1, -1, follow_symlinks=False)
        except ValueError as e:
            check(
                str(e) == "chown: cannot use fd and follow_symlinks together",
                f"chown follow_symlinks: {e}",
            )
        else:
            raise AssertionError("chown(fd, follow_symlinks=False) did not raise")
        # None and True are the defaults, not a request.
        os.chown(fd, -1, -1, dir_fd=None, follow_symlinks=True)

    if "utime" in names:
        try:
            os.utime(fd, follow_symlinks=False)
        except ValueError as e:
            check(
                str(e) == "utime: cannot use fd and follow_symlinks together",
                f"utime follow_symlinks: {e}",
            )
        else:
            raise AssertionError("utime(fd, follow_symlinks=False) did not raise")
        # 'times' and 'ns' conflict outranks the descriptor checks.
        try:
            os.utime(fd, (1, 2), ns=(1, 2), follow_symlinks=False)
        except ValueError as e:
            check(
                str(e) == "utime: you may specify either 'times' or 'ns' but not both",
                f"utime times/ns precedence: {e}",
            )
        else:
            raise AssertionError("utime times+ns did not raise")
        os.utime(fd, dir_fd=None, follow_symlinks=True)

    # ── __index__ is what makes an object a descriptor ───────────────────
    class Idx:
        def __index__(self):
            return fd

    if "statvfs" in names:
        check(os.statvfs(Idx()).f_bsize > 0, "statvfs(__index__)")
    if "chmod" in names:
        os.chmod(Idx(), 0o644)

    # ── a bad descriptor is an OSError, not a TypeError ──────────────────
    # The errno differs between platforms (EBADF and EFAULT both appear), so
    # only the class is pinned.
    # listdir and scandir are left out: -1 is the sentinel for "no descriptor"
    # (interp_posix.py:269-271 `unwrap_fd`), and where the path is nullable
    # CPython's own sentinel makes -1 indistinguishable from the omitted
    # argument, so it lists the working directory instead of reporting the bad
    # descriptor. The two references disagree, and the explicit rejection is
    # the one that cannot answer a wrong question with a right-looking list.
    for name in sorted(names & set(REST_ARGS)):
        fn = getattr(os, name)
        try:
            fn(-1, *REST_ARGS[name])
        except OSError:
            pass
        else:
            raise AssertionError(f"{name}(-1) did not raise OSError")
finally:
    os.close(fd)
    os.close(dfd)

# ── truncate on a name reports that name ────────────────────────────────
try:
    os.truncate(os.path.join(d, "nope"), 0)
except FileNotFoundError as e:
    check(e.filename == os.path.join(d, "nope"), f"truncate filename: {e.filename!r}")
else:
    raise AssertionError("truncate on a missing name did not raise")

print("OK")
