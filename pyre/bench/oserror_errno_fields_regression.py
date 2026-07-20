# A failed syscall raises the errno-specific OSError subclass with
# .errno/.strerror/.filename set and args=(errno, strerror): open() of a
# missing path -> FileNotFoundError(2), os.mkdir of an existing path ->
# FileExistsError(17).  Native-only: the wasm guest has no os/filesystem
# (open() raises NotImplementedError, `import os` has no posix backend), so
# this guard is registered with skip_backends=("wasm",).  Behaviour verified
# against CPython/PyPy.
#
# The mkdir target is a freshly created temp directory rather than "/" or the
# cwd: "/" is a Windows drive root (access-denied, not EEXIST) and the cwd is
# in-use on Windows (also access-denied), whereas a temp dir gives EEXIST on
# both POSIX and Windows.  The errno/strerror/args *values* are asserted only
# on POSIX; Windows maps different errno numbers and strerror text, so there
# the guard checks the exception type and that the fields are populated.
import os
import tempfile

POSIX = os.name == "posix"
MISSING = os.path.join(tempfile.gettempdir(), "no_such_file_xyz_pyre_probe")
EXISTING = tempfile.mkdtemp(prefix="pyre_eexist_")


def check():
    try:
        open(MISSING, "r")
    except FileNotFoundError as e:
        assert type(e).__name__ == "FileNotFoundError", type(e).__name__
        assert isinstance(e.strerror, str), e.strerror
        assert e.filename == MISSING, e.filename
        if POSIX:
            assert e.errno == 2, e.errno
            assert e.args == (2, e.strerror), e.args
        else:
            assert e.errno is not None, e.errno
    else:
        raise AssertionError("open() of a missing path did not raise")

    try:
        os.mkdir(EXISTING)
    except FileExistsError as e:
        assert type(e).__name__ == "FileExistsError", type(e).__name__
        assert isinstance(e.strerror, str), e.strerror
        if POSIX:
            assert e.errno == 17, e.errno
            assert e.args == (17, e.strerror), e.args
        else:
            assert e.errno is not None, e.errno
    else:
        raise AssertionError("os.mkdir of an existing path did not raise")


try:
    for _ in range(200):
        check()
finally:
    os.rmdir(EXISTING)
print("PASS")
