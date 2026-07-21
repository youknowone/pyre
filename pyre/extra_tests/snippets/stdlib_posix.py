import os
import posix

from testutils import assert_raises


# Sandbox builds still expose a raising `ftruncate` stub, so the capability
# bit is the check that tracks the real fd mutation.
if "HAVE_FTRUNCATE" in posix._have_functions:
    class Index:
        def __init__(self, value):
            self.value = value

        def __index__(self):
            return self.value

    class IntOnly:
        def __int__(self):
            return 0

    path = "/tmp/pyre_ftruncate_" + str(os.getpid())
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL)
    try:
        assert os.write(fd, b"abcdefgh") == 8
        os.ftruncate(Index(fd), Index(3))
        assert os.stat(path).st_size == 3
        os.ftruncate(fd, 8)
        assert os.stat(path).st_size == 8
        assert_raises(TypeError, lambda: os.ftruncate(IntOnly(), 0))
        assert_raises(TypeError, lambda: os.ftruncate(fd, IntOnly()))
        assert_raises(TypeError, lambda: os.ftruncate(fd))
        assert_raises(TypeError, lambda: os.ftruncate(fd, 0, 0))
    finally:
        os.close(fd)
        os.remove(path)
