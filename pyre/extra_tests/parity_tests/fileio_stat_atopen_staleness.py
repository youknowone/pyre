# CPython-suite gap: test_fileio never reads a file that grew after it was opened.
# parity-tests reason: PyPy answers from the descriptor, so no cached size can go stale.

"""A read answers from the descriptor, not from the file's size when it opened.

`open()` takes one `fstat` and keeps it, which is what lets a later read size
its buffer in one go and skip a second `isatty`.  Every question that snapshot
answers stops being true the moment the file changes underneath it, so each
one below changes the file after the open and asks again.
"""

import os
import sys
import tempfile

directory = tempfile.mkdtemp()
path = os.path.join(directory, "data")

with open(path, "wb") as stream:
    stream.write(b"Hello, World!")

# A second handle appends while the first is open: the first must read past
# the size its own open recorded.
with open(path, "rb") as reader:
    with open(path, "ab") as appender:
        appender.write(b" and more")
    assert reader.read() == b"Hello, World! and more"

# The same handle truncates: the recorded size now overshoots the file.
with open(path, "r+b") as stream:
    stream.truncate(5)
    stream.seek(0)
    assert stream.read() == b"Hello"

# A file that is empty at the moment it is opened records a size that says
# nothing about where its end will be, so the reader has to open first for the
# zero to be the snapshot a later read works from.
empty = os.path.join(directory, "empty")
with open(empty, "wb"):
    pass
with open(empty, "rb") as stream:
    with open(empty, "ab") as appender:
        appender.write(b"x" * 5000)
    assert stream.read() == b"x" * 5000

# A character device has no meaningful size at all, and reading one must still
# stop where it is asked to.
if os.path.exists("/dev/zero"):
    with open("/dev/zero", "rb") as stream:
        assert stream.read(8) == b"\x00" * 8

# Seekability is a property of the descriptor, so the answer may be kept, but
# it belongs to that stream alone. Only the unseekable half is platform-bound:
# a Windows anonymous-pipe descriptor answers `seekable()` True, under the
# reference CPython as much as here, so asking it there compares a backend
# against a failing reference. The seekable half below still runs everywhere.
if sys.platform != "win32":
    read_fd, write_fd = os.pipe()
    with open(read_fd, "rb") as pipe:
        assert pipe.seekable() is False
        assert pipe.seekable() is False
    os.close(write_fd)
with open(path, "rb") as stream:
    assert stream.seekable() is True
    assert stream.seekable() is True

closed = open(path, "rb")
closed.close()
try:
    closed.seekable()
except ValueError:
    pass
else:
    raise AssertionError("seekable() on a closed stream should raise")

# A regular file is never a terminal, however it is buffered.
with open(path, "r", buffering=1) as stream:
    assert stream.read() == "Hello"

print("OK")
