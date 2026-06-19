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


class StringIO:
    """In-memory text stream backed by a str buffer plus an integer
    position.  Covers the common producers/consumers (logging /
    traceback / csv / json) without the C `_io.StringIO` accelerator.
    """

    def __init__(self, initial_value="", newline="\n"):
        if newline is not None and not isinstance(newline, str):
            raise TypeError("newline must be str or None")
        if newline not in (None, "", "\n", "\r", "\r\n"):
            raise ValueError("illegal newline value: %r" % (newline,))
        self._readnl = newline
        # `newline` controls translation of '\n' on write: only '\r' and
        # '\r\n' substitute; '', '\n' and None write '\n' verbatim.
        self._writenl = newline if newline in ("\r", "\r\n") else ""
        self._readuniversal = newline is None
        self._buffer = ""
        self._pos = 0
        self._closed = False
        if initial_value is not None:
            if not isinstance(initial_value, str):
                raise TypeError("initial_value must be str or None, not %s"
                                % type(initial_value).__name__)
            self.write(initial_value)
            self._pos = 0

    def _check_closed(self):
        if self._closed:
            raise ValueError("I/O operation on closed file")

    def readable(self):
        self._check_closed()
        return True

    def writable(self):
        self._check_closed()
        return True

    def seekable(self):
        self._check_closed()
        return True

    def write(self, s):
        self._check_closed()
        if not isinstance(s, str):
            raise TypeError("string argument expected, got '%s'"
                            % type(s).__name__)
        if self._writenl:
            s = s.replace("\n", self._writenl)
        if not s:
            return 0
        pos = self._pos
        buf = self._buffer
        n = len(buf)
        if pos == n:
            self._buffer = buf + s
        elif pos > n:
            self._buffer = buf + ("\0" * (pos - n)) + s
        else:
            self._buffer = buf[:pos] + s + buf[pos + len(s):]
        self._pos = pos + len(s)
        return len(s)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def read(self, size=-1):
        self._check_closed()
        if size is None or size < 0:
            end = len(self._buffer)
        else:
            end = min(self._pos + size, len(self._buffer))
        data = self._buffer[self._pos:end]
        self._pos = end
        return data

    def readline(self, size=-1):
        self._check_closed()
        buf = self._buffer
        start = self._pos
        idx = buf.find("\n", start)
        if idx < 0:
            end = len(buf)
        else:
            end = idx + 1
        if size is not None and size >= 0:
            end = min(end, start + size)
        data = buf[start:end]
        self._pos = end
        return data

    def readlines(self, hint=-1):
        lines = []
        total = 0
        while True:
            line = self.readline()
            if not line:
                break
            lines.append(line)
            total += len(line)
            if hint is not None and hint > 0 and total >= hint:
                break
        return lines

    def seek(self, pos, whence=0):
        self._check_closed()
        if whence == 0:
            if pos < 0:
                raise ValueError("negative seek position %r" % (pos,))
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
            self._buffer = self._buffer[:size]
        return size

    def getvalue(self):
        self._check_closed()
        return self._buffer

    def flush(self):
        self._check_closed()

    @property
    def closed(self):
        return self._closed

    @property
    def line_buffering(self):
        return False

    @property
    def newlines(self):
        return None

    def close(self):
        self._closed = True
        self._buffer = ""

    def __iter__(self):
        return self

    def __next__(self):
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def __enter__(self):
        self._check_closed()
        return self

    def __exit__(self, *exc):
        self.close()
        return False
