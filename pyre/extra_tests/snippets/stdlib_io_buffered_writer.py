import io


class RawWriter(io.RawIOBase):
    def __init__(self, max_chunk=None):
        super().__init__()
        self.data = bytearray()
        self.position = 0
        self.max_chunk = max_chunk
        self.write_calls = 0

    def writable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence == 0:
            self.position = offset
        elif whence == 1:
            self.position += offset
        elif whence == 2:
            self.position = len(self.data) + offset
        return self.position

    def write(self, value):
        self.write_calls += 1
        value = bytes(value)
        if self.max_chunk is not None:
            value = value[:self.max_chunk]
        end = self.position + len(value)
        if end > len(self.data):
            self.data.extend(b"\x00" * (end - len(self.data)))
        self.data[self.position:end] = value
        self.position = end
        return len(value)

    def truncate(self, size=None):
        if size is None:
            size = self.position
        del self.data[size:]
        return size


raw = RawWriter()
writer = io.BufferedWriter(raw, buffer_size=4)
assert writer.raw is raw
assert writer.writable() is True
assert writer.seekable() is True
assert writer.write(b"abc") == 3
assert raw.data == b""
assert writer.tell() == 3
writer.flush()
assert raw.data == b"abc"

assert writer.write(memoryview(b"defgh")) == 5
assert raw.data == b"abcdefgh"
assert writer.tell() == 8
writer.flush()
assert raw.data == b"abcdefgh"

assert writer.seek(2) == 2
assert writer.write(b"XY") == 2
writer.flush()
assert raw.data == b"abXYefgh"
assert writer.truncate(5) == 5
assert raw.data == b"abXYe"

raw = RawWriter(max_chunk=2)
writer = io.BufferedWriter(raw, buffer_size=3)
assert writer.write(b"abcdefg") == 7
writer.flush()
assert raw.data == b"abcdefg"
assert raw.write_calls >= 4

raw = RawWriter()
writer = io.BufferedWriter(raw, 4)
assert writer.write(b"detach") == 6
assert writer.detach() is raw
assert raw.data == b"detach"
try:
    writer.write(b"x")
except ValueError:
    pass
else:
    raise AssertionError("detached writer operation must fail")

raw = RawWriter()
with io.BufferedWriter(raw, 4) as writer:
    assert writer.write(b"context") == 7
assert raw.closed is True
assert raw.data == b"context"


class WriterSubclass(io.BufferedWriter):
    pass


raw = RawWriter()
writer = WriterSubclass(raw)
assert type(writer) is WriterSubclass
assert writer.write(b"subclass") == 8
writer.close()
assert raw.data == b"subclass"
