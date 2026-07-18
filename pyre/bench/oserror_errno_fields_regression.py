# A failed syscall raises the errno-specific OSError subclass with
# .errno/.strerror/.filename set and args=(errno, strerror): open() of a
# missing path -> FileNotFoundError(2), os.mkdir of an existing path ->
# FileExistsError(17).  Native-only: the wasm guest has no os/filesystem
# (open() raises NotImplementedError, `import os` has no posix backend), so
# this guard is registered with skip_backends=("wasm",).  Behaviour verified
# against CPython/PyPy.
import os

PATH = "/no/such/file/xyz_pyre_probe"


def check():
    try:
        open(PATH, "r")
    except FileNotFoundError as e:
        assert type(e).__name__ == "FileNotFoundError", type(e).__name__
        assert e.errno == 2, e.errno
        assert isinstance(e.strerror, str), e.strerror
        assert e.args == (2, e.strerror), e.args
        assert e.filename == PATH, e.filename
    else:
        raise AssertionError("open() of a missing path did not raise")

    try:
        os.mkdir("/")
    except FileExistsError as e:
        assert type(e).__name__ == "FileExistsError", type(e).__name__
        assert e.errno == 17, e.errno
        assert isinstance(e.strerror, str), e.strerror
        assert e.args == (17, e.strerror), e.args
    else:
        raise AssertionError("os.mkdir('/') did not raise")


for _ in range(200):
    check()
print("PASS")
