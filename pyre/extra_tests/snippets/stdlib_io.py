import os
from io import BufferedReader, BytesIO, FileIO, RawIOBase, StringIO, TextIOWrapper

from testutils import assert_raises

fi = FileIO("README.md")
assert isinstance(fi, RawIOBase)
assert issubclass(FileIO, RawIOBase)
assert fi.mode == "rb"
assert fi.readable()
assert not fi.writable()
assert fi.seekable()
assert fi.tell() == 0
assert fi.seek(2) == 2
assert fi.tell() == 2
assert fi.seek(0) == 0
bb = BufferedReader(fi)
assert bb.seekable()

result = bb.read()

assert len(result) <= 16 * 1024
assert len(result) >= 0
assert isinstance(result, bytes)

with FileIO("README.md") as fio:
    res = fio.read()
    assert len(res) <= 16 * 1024
    assert len(res) >= 0
    assert isinstance(res, bytes)

fd = os.open("README.md", os.O_RDONLY)

with FileIO(fd) as fio:
    res2 = fio.read()
    assert res == res2

fi = FileIO("README.md")
fi.read()
fi.close()
assert fi.closefd
assert fi.closed

with assert_raises(ValueError):
    fi.read()

with FileIO("README.md") as fio:
    nres = fio.read(1)
    assert len(nres) == 1
    nres = fio.read(2)
    assert len(nres) == 2


# closefd=False leaves an externally owned descriptor usable after close.
read_fd, write_fd = os.pipe()
os.write(write_fd, b"pipe")
os.close(write_fd)
with FileIO(read_fd, closefd=False) as fio:
    assert fio.closefd is False
    assert fio.read() == b"pipe"
os.close(read_fd)

with assert_raises(ValueError):
    FileIO("README.md", closefd=False)

for bad_mode in ("", "rr", "rt", "rw", "rbb"):
    with assert_raises(ValueError):
        FileIO("README.md", bad_mode)


# Test that IOBase.isatty() raises ValueError when called on a closed file.
# Minimal subclass that inherits IOBase.isatty() without overriding it.
class MinimalRaw(RawIOBase):
    def readinto(self, b):
        return 0


f = MinimalRaw()
assert not f.closed
assert not f.isatty()  # open file: should return False

f.close()
assert f.closed

with assert_raises(ValueError):
    f.isatty()


class Gh6588:
    def __init__(self):
        self.textio = None
        self.closed = False

    def writable(self):
        return True

    def readable(self):
        return False

    def seekable(self):
        return False

    def write(self, data):
        self.textio.reconfigure(encoding="utf-8")
        return len(data)


raw = Gh6588()
textio = TextIOWrapper(raw, encoding="utf-8", write_through=True)
raw.textio = textio
with assert_raises(AttributeError):
    textio.writelines(["x"])
