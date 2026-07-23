import io

assert io.DEFAULT_BUFFER_SIZE == 128 * 1024


class ChunkRaw(io.RawIOBase):
    def __init__(self, chunks):
        super().__init__()
        self.chunks = list(chunks)
        self.position = 0
        self.read_calls = 0

    def readable(self):
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
        else:
            raise ValueError("unsupported test seek")
        return self.position

    def readinto(self, target):
        self.read_calls += 1
        if not self.chunks:
            return 0
        chunk = self.chunks.pop(0)
        if chunk is None:
            return None
        count = min(len(target), len(chunk))
        target[:count] = chunk[:count]
        if count < len(chunk):
            self.chunks.insert(0, chunk[count:])
        self.position += count
        return count


class SizingRaw(io.RawIOBase):
    def __init__(self):
        super().__init__()
        self.request_size = None

    def readable(self):
        return True

    def readinto(self, target):
        self.request_size = len(target)
        return 0


raw = SizingRaw()
assert io.BufferedReader(raw).read(1) == b""
assert raw.request_size == io.DEFAULT_BUFFER_SIZE


raw = ChunkRaw([b"abc", b"d", b"efg"])
reader = io.BufferedReader(raw, buffer_size=4)
assert reader.raw is raw
assert reader.readable() is True
assert reader.seekable() is True
assert reader.closed is False
assert reader.read(1) == b"a"
assert reader.read1(1) == b"b"
assert raw.read_calls == 1
assert reader.peek(10) == b"c"
assert raw.read_calls == 1

target = bytearray(b"xx")
assert reader.readinto(target) == 2
assert target == b"cd"
assert reader.read() == b"efg"
assert reader.read() == b""

raw = ChunkRaw([b"line 1\nline 2", b"\nend"])
reader = io.BufferedReader(raw, 8)
assert reader.readline() == b"line 1\n"
assert reader.readline() == b"line 2\n"
assert reader.readline() == b"end"

# A newline exactly at the raw chunk boundary must still terminate readline.
raw = ChunkRaw([b"abc\n", b"tail"])
reader = io.BufferedReader(raw, 4)
assert reader.readlines() == [b"abc\n", b"tail"]

# readinto1 drains readahead and permits one additional raw read.
raw = ChunkRaw([b"abc", b"def", b"ghi"])
reader = io.BufferedReader(raw, 4)
assert reader.read(1) == b"a"
target = bytearray(20)
assert reader.readinto1(target) == 5
assert target[:5] == b"bcdef"

raw = ChunkRaw([b"xyz"])
reader = io.BufferedReader(raw, 2)
assert reader.detach() is raw
for operation in (reader.read, reader.detach):
    try:
        operation()
    except ValueError:
        pass
    else:
        raise AssertionError("detached reader operation must fail")

raw = ChunkRaw([b"q"])
reader = io.BufferedReader(raw)
reader.close()
assert reader.closed is True
assert raw.closed is True
try:
    reader.read()
except ValueError:
    pass
else:
    raise AssertionError("read on a closed BufferedReader must fail")

raw = ChunkRaw([b"context"])
with io.BufferedReader(raw) as reader:
    assert reader.read() == b"context"
assert raw.closed is True


class ReaderSubclass(io.BufferedReader):
    pass


raw = ChunkRaw([b"subclass"])
reader = ReaderSubclass(raw)
assert type(reader) is ReaderSubclass
assert reader.read() == b"subclass"
