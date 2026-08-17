from testutils import assert_raises
import os
import tempfile
import warnings

fd = open("README.md")
assert "RustPython" in fd.read()

assert_raises(FileNotFoundError, open, "DoesNotExist")

# Use open as a context manager
with open("README.md", "rt") as fp:
    contents = fp.read()
    assert type(contents) == str, "type is " + str(type(contents))

with open("README.md", "r") as fp:
    contents = fp.read()
    assert type(contents) == str, "type is " + str(type(contents))

with open("README.md", "rb") as fp:
    contents = fp.read()
    assert type(contents) == bytes, "type is " + str(type(contents))

# PyPy `interp_io._open` warns before constructing `FileIO`.  When warning
# filters promote RuntimeWarning to an exception, a writable open must not
# truncate the existing file.
fd, path = tempfile.mkstemp()
try:
    os.write(fd, b"preserved")
    os.close(fd)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert_raises(RuntimeWarning, open, path, "wb", buffering=1)
    with open(path, "rb") as fp:
        assert fp.read() == b"preserved"
finally:
    os.unlink(path)


class CollectingBuffering:
    def __init__(self, calls):
        self.calls = calls

    def __index__(self):
        import gc

        self.calls.append("buffering")
        gc.collect()
        return -1


class TrackingClosefd:
    def __init__(self, calls):
        self.calls = calls

    def __bool__(self):
        self.calls.append("closefd")
        return True


class TrackingPath:
    def __init__(self, path, calls):
        self.path = path
        self.calls = calls

    def __fspath__(self):
        self.calls.append("fspath")
        return self.path


class CollectingOpener:
    def __init__(self):
        self.called = False

    def __call__(self, path, flags):
        self.called = True
        return os.open(path, flags)


# unwrap-style conversion roots every later argument before buffering can
# collect, and converts closefd before resolving the path.
fd, path = tempfile.mkstemp()
try:
    os.close(fd)
    calls = []
    opener = CollectingOpener()
    encoding = "".join(["ut", "f-8"])
    errors = "".join(["str", "ict"])
    newline = "".join(["\r", "\n"])
    with open(
        TrackingPath(path, calls),
        "r",
        buffering=CollectingBuffering(calls),
        encoding=encoding,
        errors=errors,
        newline=newline,
        closefd=TrackingClosefd(calls),
        opener=opener,
    ):
        pass
    assert calls == ["buffering", "closefd", "fspath"]
    assert opener.called
finally:
    os.unlink(path)


class RaisingClosefd(TrackingClosefd):
    def __bool__(self):
        self.calls.append("closefd")
        raise RuntimeError("closefd conversion ran first")


calls = []
assert_raises(
    RuntimeError,
    open,
    TrackingPath("unused", calls),
    closefd=RaisingClosefd(calls),
)
assert calls == ["closefd"]


class RaisingPath:
    def __fspath__(self):
        raise LookupError("path resolution ran first")


with warnings.catch_warnings():
    warnings.simplefilter("error", RuntimeWarning)
    assert_raises(LookupError, open, RaisingPath(), "wb", buffering=1)

# PyPy's unwrap_spec converts later arguments before `_open` resolves PathLike.
assert_raises(TypeError, open, RaisingPath(), object())


class DescriptorAndPath:
    def __init__(self, descriptor, path):
        self.descriptor = descriptor
        self.path = path
        self.calls = []

    def __index__(self):
        self.calls.append("__index__")
        return self.descriptor

    def __fspath__(self):
        self.calls.append("__fspath__")
        return self.path


# `_open` never probes `__index__` for a non-int `file` argument: PyPy routes
# every non-str/bytes/int argument straight to `fspath`
# (pypy/module/_io/interp_io.py:36), matching the pinned CPython fallback
# (lib-python/3/_pyio.py:194, `if not isinstance(file, int): file =
# os.fspath(file)`). An object implementing both protocols resolves through
# __fspath__ only, and the resolved path becomes the public name.
fd, path = tempfile.mkstemp()
os.close(fd)
try:
    target = DescriptorAndPath(fd, path)
    with open(target, "rb") as fp:
        assert fp.name == path
    assert target.calls == ["__fspath__"]
finally:
    os.unlink(path)


class CollectingPath:
    def __init__(self, path):
        self.path = path

    def __fspath__(self):
        import gc

        gc.collect()
        return self.path


# Path conversion may trigger a moving collection; later open arguments must
# remain rooted until the constructors consume them.
fd, path = tempfile.mkstemp()
try:
    os.close(fd)
    opener = CollectingOpener()
    encoding = "".join(["ut", "f-8"])
    errors = "".join(["str", "ict"])
    newline = "".join(["\r", "\n"])
    with open(
        CollectingPath(path),
        "r",
        encoding=encoding,
        errors=errors,
        newline=newline,
        opener=opener,
    ):
        pass
    assert opener.called
finally:
    os.unlink(path)
