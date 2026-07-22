from io import BufferedRWPair, RawIOBase

from testutils import assert_raises


class Reader(RawIOBase):
    def __init__(self, data=b"", isatty=False):
        self.data = bytearray(data)
        self.pos = 0
        self.is_a_tty = isatty

    def readable(self):
        return True

    def readinto(self, target):
        size = min(len(target), len(self.data) - self.pos)
        target[:size] = self.data[self.pos : self.pos + size]
        self.pos += size
        return size

    def isatty(self):
        return self.is_a_tty


class Writer(RawIOBase):
    def __init__(self, isatty=False):
        self.output = bytearray()
        self.is_a_tty = isatty

    def writable(self):
        return True

    def write(self, data):
        self.output.extend(data)
        return len(data)

    def isatty(self):
        return self.is_a_tty


pair = BufferedRWPair(Reader(b"abcdef"), Writer(), 4)
assert pair.read(3) == b"abc"
assert pair.read1(2) == b"d"
assert pair.peek(1).startswith(b"e")

target = bytearray(2)
assert pair.readinto(target) == 2
assert target == b"ef"

pair = BufferedRWPair(Reader(b"abcdef"), Writer(), 4)
target = bytearray(5)
assert pair.readinto1(target) == 5
assert target == b"abcde"

writer = Writer()
pair = BufferedRWPair(Reader(), writer, 4)
assert pair.write(b"abc") == 3
source = bytearray(b"def")
assert pair.write(source) == 3
source[:] = b"***"
pair.flush()
assert writer.output == b"abcdef"

assert pair.readable()
assert pair.writable()
assert not pair.seekable()
assert not pair.closed
pair.close()
assert pair.closed

assert not BufferedRWPair(Reader(), Writer()).isatty()
assert BufferedRWPair(Reader(isatty=True), Writer()).isatty()
assert BufferedRWPair(Reader(), Writer(isatty=True)).isatty()

uninitialized = BufferedRWPair.__new__(BufferedRWPair)
with assert_raises(ValueError):
    uninitialized.read(0)
with assert_raises(ValueError):
    uninitialized.write(b"")
uninitialized.__init__(Reader(), Writer())
assert uninitialized.read(0) == b""
assert uninitialized.write(b"") == 0


class NotReadable(Reader):
    def readable(self):
        return False


class NotWritable(Writer):
    def writable(self):
        return False


with assert_raises(OSError):
    BufferedRWPair(NotReadable(), Writer())
with assert_raises(OSError):
    BufferedRWPair(Reader(), NotWritable())
with assert_raises(TypeError):
    BufferedRWPair(Reader(), Writer(), 8, 12)


class CloseErrorReader(Reader):
    def close(self):
        raise NameError("reader close")


class CloseErrorWriter(Writer):
    def close(self):
        raise NameError("writer close")


reader = Reader()
writer = CloseErrorWriter()
pair = BufferedRWPair(reader, writer)
with assert_raises(NameError):
    pair.close()
assert reader.closed
writer.close = lambda: None
pair.__init__(Reader(), Writer())
pair.close()

reader = CloseErrorReader()
writer = Writer()
pair = BufferedRWPair(reader, writer)
with assert_raises(NameError):
    pair.close()
assert writer.closed
reader.close = lambda: None
pair.__init__(Reader(), Writer())
pair.close()

reader = CloseErrorReader()
writer = CloseErrorWriter()
pair = BufferedRWPair(reader, writer)
try:
    pair.close()
except NameError as error:
    assert str(error) == "reader close"
    assert isinstance(error.__context__, NameError)
    assert str(error.__context__) == "writer close"
else:
    raise AssertionError("reader close error was not raised")
reader.close = lambda: None
writer.close = lambda: None
pair.__init__(Reader(), Writer())
pair.close()
