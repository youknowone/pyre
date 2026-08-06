"""A failed filesystem call reports the code the platform gave it.

Windows names its failures with `GetLastError` codes rather than errnos, and
`PyErr_SetExcFromWindowsErrWithFilenameObjects` keeps that code: `winerror`
holds it, `strerror` is the system's own message for it, and `errno` — with the
subclass chosen from it — comes from the `rwin32.py:285-306` map, so `str(e)`
opens with `[WinError 3]` and not `[Errno 2]`.  The descriptor calls go through
the C runtime, which reports an errno, and those keep the errno spelling on
every platform.

The message text is the operating system's and is localized, so only its
presence is asserted.
"""

import os
import sys
import tempfile

WIN32 = sys.platform == "win32"

base = tempfile.mkdtemp(prefix="pyre_oserr_")
a_dir = os.path.join(base, "d")
os.mkdir(a_dir)
a_file = os.path.join(base, "f")
with open(a_file, "w") as fp:
    fp.write("x")
missing = os.path.join(base, "missing")
missing_deep = os.path.join(base, "missing", "deeper")


def failure(fn, *args):
    try:
        fn(*args)
    except OSError as exc:
        return exc
    raise AssertionError("%s%r must fail" % (fn, args))


def check(exc, cls, errno, winerror, filename, filename2=None):
    assert type(exc) is cls, (type(exc), cls)
    assert exc.errno == errno, (exc.errno, errno)
    assert exc.filename == filename, (exc.filename, filename)
    assert exc.filename2 == filename2, (exc.filename2, filename2)
    # The message is the platform's; it is a non-empty string and the whole of
    # `args[1]`, whatever it says.
    assert isinstance(exc.strerror, str) and exc.strerror, exc.strerror
    assert exc.args == (errno, exc.strerror), exc.args
    if WIN32:
        assert exc.winerror == winerror, (exc.winerror, winerror)
        expected = "[WinError %d] %s" % (winerror, exc.strerror)
    else:
        assert not hasattr(exc, "winerror")
        expected = "[Errno %d] %s" % (errno, exc.strerror)
    if filename is not None:
        expected += ": %r" % (filename,)
        if filename2 is not None:
            expected += " -> %r" % (filename2,)
    assert str(exc) == expected, (str(exc), expected)


# `ERROR_FILE_NOT_FOUND` and `ERROR_PATH_NOT_FOUND` are separate codes that
# share `ENOENT`: which one arrives says whether the leaf or a parent is the
# missing component.
check(failure(os.stat, missing), FileNotFoundError, 2, 2, missing)
check(failure(os.stat, missing_deep), FileNotFoundError, 2, 3, missing_deep)
check(failure(os.lstat, missing), FileNotFoundError, 2, 2, missing)
check(failure(os.listdir, missing), FileNotFoundError, 2, 3, missing)
check(failure(lambda p: list(os.scandir(p)), missing), FileNotFoundError, 2, 3, missing)
check(failure(os.mkdir, missing_deep), FileNotFoundError, 2, 3, missing_deep)
check(failure(os.rmdir, missing), FileNotFoundError, 2, 2, missing)
check(failure(os.remove, missing), FileNotFoundError, 2, 2, missing)
check(failure(os.unlink, missing), FileNotFoundError, 2, 2, missing)
# `os_utime_impl` is the one path call whose POSIX arm leaves the name out of
# the error it raises; the Windows arm names the file like the rest of them.
check(failure(os.utime, missing), FileNotFoundError, 2, 2, missing if WIN32 else None)

check(failure(os.mkdir, a_dir), FileExistsError, 17, 183, a_dir)
check(failure(os.listdir, a_file), NotADirectoryError, 20, 267, a_file)
check(failure(os.rmdir, a_file), NotADirectoryError, 20, 267, a_file)

# A code outside the map keeps the plain `OSError` and takes `EINVAL`, which is
# the errno POSIX reports for reading a link that is not one.  Which code the
# filesystem answers a non-reparse point with is its own business, so only its
# flavour is asserted.
exc = failure(os.readlink, a_file)
assert type(exc) is OSError, type(exc)
assert exc.filename == a_file, exc.filename
if WIN32:
    assert exc.winerror is not None, exc.winerror
    assert str(exc).startswith("[WinError %d] " % exc.winerror), str(exc)
else:
    assert exc.errno == 22, exc.errno

# Both paths of a two-path call are reported, and the arrow between them is the
# same either way the code is spelled.
moved = os.path.join(base, "moved")
check(failure(os.rename, missing, moved), FileNotFoundError, 2, 2, missing, moved)
check(failure(os.replace, missing, moved), FileNotFoundError, 2, 2, missing, moved)

# `os.open` is the C runtime's, which reports an errno on every platform.
exc = failure(os.open, missing_deep, os.O_RDONLY)
assert type(exc) is FileNotFoundError, type(exc)
assert exc.errno == 2, exc.errno
assert exc.args == (2, exc.strerror), exc.args
assert str(exc) == "[Errno 2] %s: %r" % (exc.strerror, missing_deep), str(exc)
if WIN32:
    assert exc.winerror is None, exc.winerror

# So are the descriptor calls: a descriptor that names nothing is `EBADF`, and
# on Windows no Win32 code comes with it.  (The C runtime's own answer to one
# is to abort the process, which is why the calls silence its invalid parameter
# handler first.)
BAD_FD = 999
# The loop closes, truncates and writes to this descriptor, so the run has to
# start by establishing that it names nothing.
try:
    os.fstat(BAD_FD)
except OSError:
    pass
else:
    raise AssertionError("fd %d is open; the loop below would write to it" % BAD_FD)
for call, args in (
    (os.close, ()),
    (os.dup, ()),
    (os.lseek, (0, os.SEEK_SET)),
    (os.read, (4,)),
    (os.write, (b"x",)),
    (os.fsync, ()),
    (os.ftruncate, (0,)),
    (os.dup2, (BAD_FD - 1,)),
):
    exc = failure(call, BAD_FD, *args)
    assert exc.errno == 9, (call, exc.errno)
    assert str(exc) == "[Errno 9] %s" % exc.strerror, (call, str(exc))
    if WIN32:
        assert exc.winerror is None, (call, exc.winerror)

# `os.fstat` is the exception: it reaches for the descriptor's handle, and the
# Win32 error for the one it does not have is what gets reported.
exc = failure(os.fstat, BAD_FD)
assert exc.errno == 9, exc.errno
if WIN32:
    assert exc.winerror == 6, exc.winerror
    assert str(exc) == "[WinError 6] %s" % exc.strerror, str(exc)
else:
    assert str(exc) == "[Errno 9] %s" % exc.strerror, str(exc)

# A descriptor that names nothing is not a terminal, which is reported rather
# than raised.
assert os.isatty(BAD_FD) is False

# `os.startfile` is Windows' alone: it hands the file to whatever program is
# registered for it, and a name no file answers to never reaches one.
if WIN32:
    check(failure(os.startfile, missing), FileNotFoundError, 2, 2, missing)
    check(failure(os.startfile, missing, "open"), FileNotFoundError, 2, 2, missing)
    # Its four optional arguments are positional-or-keyword, and the keyword
    # spelling is the one `webbrowser` and `os.startfile(..., cwd=)` callers
    # use — a call that drops them opens the file in the wrong directory.
    check(
        failure(
            lambda p: os.startfile(p, operation="open", arguments="", cwd=base, show_cmd=0),
            missing,
        ),
        FileNotFoundError,
        2,
        2,
        missing,
    )
    try:
        os.startfile(missing, bogus=1)
    except TypeError as exc:
        assert "unexpected keyword argument 'bogus'" in str(exc), str(exc)
    else:
        raise AssertionError("startfile must reject an unknown keyword")
    try:
        os.startfile()
    except TypeError:
        pass
    else:
        raise AssertionError("startfile must require a file")
else:
    assert not hasattr(os, "startfile")

os.remove(a_file)
os.rmdir(a_dir)
os.rmdir(base)
print("OK")
