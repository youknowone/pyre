"""App-level fallbacks for the _io module.

BytesIO is an in-memory binary stream backed by a bytearray plus an
integer position, sufficient for pickle's pure-Python Pickler/Unpickler
(write/getvalue on dump, read/readline on load).
"""


class BytesIO:
    def __init__(self, initial_bytes=b""):
        self._buffer = bytearray(initial_bytes)
        self._pos = 0
        self._closed = False

    def _check_closed(self):
        if self._closed:
            raise ValueError("I/O operation on closed file.")

    def readable(self):
        self._check_closed()
        return True

    def writable(self):
        self._check_closed()
        return True

    def seekable(self):
        self._check_closed()
        return True

    def read(self, size=-1):
        self._check_closed()
        if size is None or size < 0:
            end = len(self._buffer)
        else:
            end = min(self._pos + size, len(self._buffer))
        data = bytes(self._buffer[self._pos:end])
        self._pos = end
        return data

    def read1(self, size=-1):
        return self.read(size)

    def readline(self, size=-1):
        self._check_closed()
        buf = self._buffer
        n = len(buf)
        start = self._pos
        idx = buf.find(b"\n", start)
        if idx < 0:
            end = n
        else:
            end = idx + 1
        if size is not None and size >= 0:
            end = min(end, start + size)
        data = bytes(buf[start:end])
        self._pos = end
        return data

    def readlines(self, hint=-1):
        lines = []
        total = 0
        while True:
            line = self.readline()
            if len(line) == 0:
                break
            lines.append(line)
            total += len(line)
            if hint is not None and hint > 0 and total >= hint:
                break
        return lines

    def write(self, b):
        self._check_closed()
        data = bytes(b)
        pos = self._pos
        buf = self._buffer
        n = len(buf)
        if pos == n:
            # Append — the common path (pickle always writes at the end).
            buf.extend(data)
        else:
            if pos > n:
                buf.extend(b"\x00" * (pos - n))
            # Overwrite/extend without slice assignment (STORE_SLICE).
            head = bytes(buf[:pos])
            self._buffer = bytearray(head)
            self._buffer.extend(data)
            tail_start = pos + len(data)
            if tail_start < n:
                self._buffer.extend(bytes(buf[tail_start:n]))
            buf = self._buffer
        self._pos = pos + len(data)
        return len(data)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def seek(self, pos, whence=0):
        self._check_closed()
        if whence == 0:
            if pos < 0:
                raise ValueError("negative seek value %r" % (pos,))
            newpos = pos
        elif whence == 1:
            newpos = self._pos + pos
        elif whence == 2:
            newpos = len(self._buffer) + pos
        else:
            raise ValueError("invalid whence (%r, should be 0, 1 or 2)" % (whence,))
        if newpos < 0:
            newpos = 0
        self._pos = newpos
        return newpos

    def tell(self):
        self._check_closed()
        return self._pos

    def truncate(self, size=None):
        self._check_closed()
        if size is None:
            size = self._pos
        if size < 0:
            raise ValueError("negative truncate size %r" % (size,))
        if size < len(self._buffer):
            self._buffer = bytearray(bytes(self._buffer[:size]))
        return size

    def getvalue(self):
        self._check_closed()
        return bytes(self._buffer)

    def getbuffer(self):
        self._check_closed()
        return memoryview(self._buffer)

    def flush(self):
        self._check_closed()

    @property
    def closed(self):
        return self._closed

    def close(self):
        self._closed = True
        self._buffer = bytearray()

    def __iter__(self):
        return self

    def __next__(self):
        line = self.readline()
        if len(line) == 0:
            raise StopIteration
        return line

    def __enter__(self):
        self._check_closed()
        return self

    def __exit__(self, *exc):
        self.close()
        return False
