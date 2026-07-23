from io import BufferedRandom, RawIOBase

from testutils import assert_raises


class RandomRaw(RawIOBase):
    def __init__(self, data=b""):
        self.data = bytearray(data)
        self.pos = 0
        self.flushes = 0

    def readable(self):
        return True

    def writable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence == 0:
            new_pos = offset
        elif whence == 1:
            new_pos = self.pos + offset
        elif whence == 2:
            new_pos = len(self.data) + offset
        else:
            raise ValueError("bad whence")
        if new_pos < 0:
            raise OSError("negative seek")
        self.pos = new_pos
        return new_pos

    def readinto(self, target):
        size = min(len(target), max(0, len(self.data) - self.pos))
        target[:size] = self.data[self.pos : self.pos + size]
        self.pos += size
        return size

    def read(self, size=-1):
        if size < 0:
            size = len(self.data) - self.pos
        result = bytes(self.data[self.pos : self.pos + size])
        self.pos += len(result)
        return result

    def write(self, source):
        end = self.pos + len(source)
        if end > len(self.data):
            self.data.extend(b"\0" * (end - len(self.data)))
        self.data[self.pos : end] = source
        self.pos = end
        return len(source)

    def truncate(self, size=None):
        if size is None:
            size = self.pos
        del self.data[size:]
        return size

    def flush(self):
        self.flushes += 1


raw = RandomRaw(b"abcdefgh")
stream = BufferedRandom(raw, 4)
assert stream.raw is raw
assert stream.readable() and stream.writable() and stream.seekable()
assert stream.read(2) == b"ab"
assert stream.tell() == 2

# A write after readahead must rewind to the logical, not raw, position.
assert stream.write(b"XY") == 2
assert stream.tell() == 4
stream.flush()
assert raw.data == b"abXYefgh"
assert raw.pos == 4

# Reading after a pending write flushes it and resumes at its logical end.
assert stream.write(b"12") == 2
assert stream.read(2) == b"gh"
assert raw.data == b"abXY12gh"
assert stream.seek(0) == 0
assert stream.read() == b"abXY12gh"

stream.seek(1)
target = bytearray(3)
assert stream.readinto(target) == 3
assert target == b"bXY"
stream.seek(0)
target = bytearray(5)
assert stream.readinto1(target) == 5
assert target == b"abXY1"

stream.seek(0, 2)
assert stream.write(b"!") == 1
assert stream.truncate() == 9
stream.flush()
assert raw.data == b"abXY12gh!"

stream.close()
assert stream.closed
assert raw.closed
with assert_raises(ValueError):
    stream.read(1)

raw = RandomRaw(b"detach")
stream = BufferedRandom(raw, 2)
assert stream.write(b"DE") == 2
assert stream.detach() is raw
assert raw.data == b"DEtach"
with assert_raises(ValueError):
    stream.read(1)

uninitialized = BufferedRandom.__new__(BufferedRandom)
with assert_raises(ValueError):
    uninitialized.read(1)
uninitialized.__init__(RandomRaw(b"ok"))
assert uninitialized.read() == b"ok"
uninitialized.close()


class NotReadable(RandomRaw):
    def readable(self):
        return False


class NotWritable(RandomRaw):
    def writable(self):
        return False


class NotSeekable(RandomRaw):
    def seekable(self):
        return False


with assert_raises(OSError):
    BufferedRandom(NotReadable())
with assert_raises(OSError):
    BufferedRandom(NotWritable())
with assert_raises(OSError):
    BufferedRandom(NotSeekable())
with assert_raises(ValueError):
    BufferedRandom(RandomRaw(), 0)
with assert_raises(TypeError):
    BufferedRandom(RandomRaw(), 8, 12)
