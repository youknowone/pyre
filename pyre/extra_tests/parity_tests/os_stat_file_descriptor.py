# pyre-check: platforms=linux,darwin
# Windows has no `dir_fd`, so the reference raises NotImplementedError where
# this pins the ValueError that precedes it. pyre reads EBADF from
# `os.stat(fd)` there as well, which this file cannot gate: a comparison
# against a failing reference measures nothing.
"""`os.stat` takes an open file descriptor where `os.lstat` does not.

`interp_posix.py:611` declares stat's path as `path_or_fd(allow_fd=True)` and
`:659` declares lstat's as `allow_fd=False`, so the descriptor form belongs to
one of them only — and that difference is also what makes their type errors name
different allowed types.

`os.stat in os.supports_fd` is unconditionally true (`os.py:148`, "fstat always
works"), so this is the capability the set has always advertised.

`do_stat` (`interp_posix.py:634-644`) tests the descriptor before anything else:
holding one, neither `dir_fd` nor `follow_symlinks` has a path to apply to, and
both rejections come before the platform's `dir_fd` availability is consulted.
"""

import os
import tempfile
import warnings

assert os.stat in os.supports_fd, "os.stat has always been advertised as fd-capable"

tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "f")
with open(path, "wb") as f:
    f.write(b"0123456789")

fd = os.open(path, os.O_RDONLY)
try:
    by_fd = os.stat(fd)
    by_path = os.stat(path)
    assert by_fd.st_size == 10, by_fd.st_size
    # The same file either way: the descriptor form is `fstat`, not a re-open.
    assert (by_fd.st_ino, by_fd.st_dev) == (by_path.st_ino, by_path.st_dev)
    assert os.stat(fd) == os.fstat(fd)

    # `True` is an `int`, so it names descriptor 1. Whether a bool used as a
    # descriptor warns is a separate question; the value is what matters here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert os.stat(True).st_dev == os.fstat(1).st_dev

    # Neither other argument has anything to apply to.
    try:
        os.stat(fd, dir_fd=fd)
    except ValueError as exc:
        assert str(exc) == "stat: can't specify dir_fd without matching path", str(exc)
    else:
        raise AssertionError("stat accepted dir_fd with a descriptor")

    try:
        os.stat(fd, follow_symlinks=False)
    except ValueError as exc:
        assert str(exc) == "stat: cannot use fd and follow_symlinks together", str(exc)
    else:
        raise AssertionError("stat accepted follow_symlinks with a descriptor")

    # lstat takes no descriptor, and says so with its own name and its own
    # allowed-type list.
    try:
        os.lstat(fd)
    except TypeError as exc:
        assert str(exc) == "lstat: path should be string, bytes or os.PathLike, not int", str(exc)
    else:
        raise AssertionError("lstat accepted a descriptor")

    # The widened list appears only where the descriptor is allowed.
    try:
        os.stat(1.5)
    except TypeError as exc:
        expected = "stat: path should be string, bytes, os.PathLike or integer, not float"
        assert str(exc) == expected, str(exc)
    else:
        raise AssertionError("stat accepted a float")

    try:
        os.lstat(1.5)
    except TypeError as exc:
        expected = "lstat: path should be string, bytes or os.PathLike, not float"
        assert str(exc) == expected, str(exc)
    else:
        raise AssertionError("lstat accepted a float")

    # The arguments are unwrapped in signature order, so with more than one of
    # them bad it is the leftmost that answers. Each can also run user code —
    # `__fspath__`, `__index__`, `__bool__` — so the order is observable even
    # when nothing raises.
    try:
        os.stat(1.5, dir_fd=1.5)
    except TypeError as exc:
        expected = "stat: path should be string, bytes, os.PathLike or integer, not float"
        assert str(exc) == expected, str(exc)
    else:
        raise AssertionError("stat accepted a float path")

    order = []

    class Spy:
        def __fspath__(self):
            order.append("path")
            return tmp

    class Truthy:
        def __bool__(self):
            order.append("follow_symlinks")
            return True

    try:
        os.stat(Spy(), dir_fd=1.5, follow_symlinks=Truthy())
    except TypeError as exc:
        assert str(exc) == "argument should be integer or None, not float", str(exc)
    else:
        raise AssertionError("stat accepted a float dir_fd")
    # `path` was resolved, `dir_fd` then rejected, `follow_symlinks` never read.
    assert order == ["path"], order

    # The descriptor probe is `__index__`, and an object carrying both it and
    # `__fspath__` is taken as a descriptor — so an `__index__` that raises
    # reports its own exception instead of falling through to the path.
    class BadIndex:
        def __index__(self):
            raise RuntimeError("boom")

        def __fspath__(self):
            return path

    try:
        os.stat(BadIndex())
    except RuntimeError as exc:
        assert str(exc) == "boom", str(exc)
    else:
        raise AssertionError("stat fell through a raising __index__ to __fspath__")

    # lstat takes no descriptor, so it never probes __index__ at all.
    assert os.lstat(BadIndex()).st_size == 10

    # A descriptor no call can serve reports the descriptor, not the type: -1
    # is an OSError either way. Which errno it carries is a property of the
    # libc path taken and differs between the two entry points even upstream
    # (EFAULT from `stat`, EBADF from `fstat`), so only the class is pinned.
    try:
        os.stat(-1)
    except OSError:
        pass
    else:
        raise AssertionError("os.stat(-1) did not fail")
finally:
    os.close(fd)

# `dir_fd` is a separate capability, reported honestly: a platform that does
# not honour it says so, and a platform that does resolves a relative name
# against the descriptor. `os.py:120-121` reads HAVE_FSTATAT for `stat` and
# HAVE_LSTAT for `lstat`, so the two are advertised independently.
if os.stat in os.supports_dir_fd:
    assert os.lstat in os.supports_dir_fd, "lstat takes dir_fd wherever stat does"
    link = os.path.join(tmp, "link")
    os.symlink("f", link)
    dfd = os.open(tmp, os.O_RDONLY)
    try:
        # `fstatat` resolves the name against the descriptor, and reaches the
        # same file the path form does.
        by_dir_fd = os.stat("f", dir_fd=dfd)
        assert by_dir_fd.st_size == 10, by_dir_fd.st_size
        assert (by_dir_fd.st_ino, by_dir_fd.st_dev) == (by_path.st_ino, by_path.st_dev)

        # An absolute name ignores the descriptor entirely.
        assert os.stat(path, dir_fd=dfd).st_ino == by_path.st_ino

        # AT_SYMLINK_NOFOLLOW is what carries follow_symlinks=False, so the
        # two spellings of "do not follow" agree.
        nofollow = os.stat("link", dir_fd=dfd, follow_symlinks=False)
        assert nofollow.st_ino == os.lstat("link", dir_fd=dfd).st_ino
        assert nofollow.st_ino == os.lstat(link).st_ino
        assert nofollow.st_ino != by_path.st_ino, "lstat followed the symlink"
        assert os.stat("link", dir_fd=dfd).st_ino == by_path.st_ino

        # A missing name reports the name, not the descriptor.
        try:
            os.stat("absent", dir_fd=dfd)
        except FileNotFoundError as exc:
            assert exc.filename == "absent", exc.filename
        else:
            raise AssertionError("stat found a name that does not exist")

        # A descriptor that is not a directory cannot resolve a relative name.
        plain = os.open(path, os.O_RDONLY)
        try:
            os.stat("f", dir_fd=plain)
        except NotADirectoryError:
            pass
        else:
            raise AssertionError("stat resolved a name against a plain file")
        finally:
            os.close(plain)

        # `_unwrap_dirfd` types the argument before it reaches the syscall.
        try:
            os.stat("f", dir_fd=1.5)
        except TypeError as exc:
            assert str(exc) == "argument should be integer or None, not float", str(exc)
        else:
            raise AssertionError("stat accepted a float dir_fd")

        # A descriptor no call can serve is an OSError. Which errno it carries
        # depends on where the rejection happens — the sentinel check or the
        # syscall — so only the class is pinned.
        try:
            os.stat("f", dir_fd=-1)
        except OSError:
            pass
        else:
            raise AssertionError("stat accepted dir_fd=-1")
    finally:
        os.close(dfd)

print("OK")
