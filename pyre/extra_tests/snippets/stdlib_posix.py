import gc
import shutil
import tempfile
import warnings
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


# `__fspath__` is looked up on the type and its descriptor is resolved against
# the instance, so a property runs and what it answers is what gets called.  A
# `None` left on the type switches the protocol off the way `__hash__ = None`
# does, and the object is reported as not path-like rather than as something
# that failed to be called.
class FsPathOk:
    def __fspath__(self):
        return "fspath-ok"


class FsPathNone:
    __fspath__ = None


class FsPathNoneOverride(FsPathOk):
    __fspath__ = None


class FsPathProperty:
    __fspath__ = property(lambda self: (lambda: "fspath-property"))


class FsPathPropertyNone:
    __fspath__ = property(lambda self: None)


assert os.fspath(FsPathOk()) == "fspath-ok"
assert os.fspath(FsPathProperty()) == "fspath-property"

for cls in (FsPathNone, FsPathNoneOverride, FsPathPropertyNone):
    try:
        os.fspath(cls())
    except TypeError as exc:
        # The object is named, not the `None` that was found in its place.
        assert cls.__name__ in str(exc), (cls, exc)
        assert "NoneType" not in str(exc), (cls, exc)
    else:
        raise AssertionError("os.fspath(%s()) did not raise" % cls.__name__)

    # The path converter every other entry point shares reaches the same
    # decision, under its own wording.
    try:
        open(cls())
    except TypeError as exc:
        assert cls.__name__ in str(exc), (cls, exc)
        assert "NoneType" not in str(exc), (cls, exc)
    else:
        raise AssertionError("open(%s()) did not raise" % cls.__name__)

    try:
        os.rename(cls(), "unused")
    except TypeError as exc:
        assert cls.__name__ in str(exc), (cls, exc)
        assert "NoneType" not in str(exc), (cls, exc)
    else:
        raise AssertionError("os.rename(%s(), ...) did not raise" % cls.__name__)


# The user and group id setters convert through the `uid_t` unwrapper, so a
# non-integer and a value past the 32-bit range are turned away before any
# privilege change is attempted.  Only the rejected forms are exercised here:
# every call below raises during conversion and reaches no syscall.
for _name in ("setuid", "seteuid", "setgid", "setegid"):
    _setter = getattr(os, _name, None)
    if _setter is None:
        continue
    assert_raises(TypeError, lambda f=_setter: f("not an int"))
    assert_raises(OverflowError, lambda f=_setter: f(1 << 32))

for _name in ("setreuid", "setregid"):
    _setter = getattr(os, _name, None)
    if _setter is None:
        continue
    assert_raises(TypeError, lambda f=_setter: f("not an int", 0))
    assert_raises(TypeError, lambda f=_setter: f(0, "not an int"))
    assert_raises(OverflowError, lambda f=_setter: f(1 << 32, 0))
    assert_raises(OverflowError, lambda f=_setter: f(0, 1 << 32))


# An iterator from `os.scandir` that was neither closed nor run to the end is
# an unclosed one, and says so when it is collected.  Closing it, or reaching
# the end of the enumeration, is what makes it silent.
def _scandir_warnings(use):
    directory = tempfile.mkdtemp()
    try:
        for name in ("a", "b"):
            with open(os.path.join(directory, name), "w"):
                pass
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            use(directory)
            gc.collect()
        return [w for w in caught if issubclass(w.category, ResourceWarning)]
    finally:
        shutil.rmtree(directory)


def _abandon_midway(directory):
    iterator = os.scandir(directory)
    next(iterator)
    del iterator


def _exhaust(directory):
    iterator = os.scandir(directory)
    list(iterator)
    del iterator


def _close_midway(directory):
    with os.scandir(directory) as iterator:
        next(iterator)
    del iterator


assert _scandir_warnings(_abandon_midway), "an abandoned scandir iterator is unclosed"
assert not _scandir_warnings(_exhaust), "an exhausted scandir iterator is closed"
assert not _scandir_warnings(_close_midway), "a closed scandir iterator is closed"
