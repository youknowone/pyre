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

    def _read_from_buffer(self, size=-1):
        # `W_BytesIO.read_w` copies straight out of the internal buffer, so the
        # sibling slots below reach it here rather than through `self.read`,
        # which a subclass may override.
        self._check_closed()
        if size is None or size < 0:
            end = len(self._buffer)
        else:
            end = min(self._pos + size, len(self._buffer))
        data = bytes(self._buffer[self._pos:end])
        self._pos = end
        return data

    def read(self, size=-1):
        return self._read_from_buffer(size)

    def read1(self, size=-1):
        return self._read_from_buffer(size)

    def _readinto_from_buffer(self, buffer):
        self._check_closed()
        # `W_BytesIO.readinto_w`: acquire a writable buffer for the
        # duration of the read, consume at most its byte length, copy the
        # output at offset zero, and return the number of bytes copied.
        with memoryview(buffer) as view:
            if view.readonly:
                raise TypeError("readinto() argument must be read-write bytes-like object")
            target = view.cast("B")
            output = self._read_from_buffer(target.nbytes)
            target[:len(output)] = output
            return len(output)

    def readinto(self, buffer):
        return self._readinto_from_buffer(buffer)

    def readinto1(self, buffer):
        return self._readinto_from_buffer(buffer)

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


class IncrementalNewlineDecoder:
    r"""Codec used when reading a file in universal newlines mode.  It wraps
    another incremental decoder, translating \r\n and \r into \n.  It also
    records the types of newlines encountered.  When used with
    translate=False, it ensures that the newline sequence is returned in
    one piece.

    `_io.IncrementalNewlineDecoder` is a standalone type, not a
    `codecs.IncrementalDecoder` subclass; `decode_source` and `TextIOWrapper`
    construct it with `decoder=None` to translate an already-decoded string.
    """

    _LF = 1
    _CR = 2
    _CRLF = 4

    def __init__(self, decoder, translate, errors="strict"):
        if errors is None:
            errors = "strict"
        elif not isinstance(errors, str):
            raise TypeError(
                "TextIOWrapper() argument 'errors' must be str or None, not %s"
                % type(errors).__name__
            )
        else:
            # io_check_errors minus the dev-mode handler lookup — a codecs
            # import here would recurse through decode_source.
            errors.encode("utf-8", "strict")
        if not isinstance(translate, int):
            try:
                translate = translate.__index__()
            except AttributeError:
                raise TypeError(
                    "'%s' object cannot be interpreted as an integer"
                    % type(translate).__name__
                ) from None
        self.errors = errors
        self.translate = translate
        self.decoder = decoder
        self.seennl = 0
        self.pendingcr = False

    def decode(self, input, final=False):
        # decode input (with the eventual \r from a previous pass)
        if self.decoder is None:
            output = input
        else:
            output = self.decoder.decode(input, final)
        if not isinstance(output, str):
            raise TypeError("decoder should return a string result")
        if self.pendingcr and (output or final):
            output = "\r" + output
            self.pendingcr = False

        # retain last \r even when not translating data:
        # then readline() is sure to get \r\n in one pass
        if output.endswith("\r") and not final:
            output = output[:-1]
            self.pendingcr = True

        # Record which newlines are read
        crlf = output.count("\r\n")
        cr = output.count("\r") - crlf
        lf = output.count("\n") - crlf
        self.seennl |= (
            (lf and self._LF) | (cr and self._CR) | (crlf and self._CRLF)
        )

        if self.translate:
            if crlf:
                output = output.replace("\r\n", "\n")
            if cr:
                output = output.replace("\r", "\n")

        return output

    def getstate(self):
        if self.decoder is None:
            buf = b""
            flag = 0
        else:
            buf, flag = self.decoder.getstate()
        flag <<= 1
        if self.pendingcr:
            flag |= 1
        return buf, flag

    def setstate(self, state):
        buf, flag = state
        self.pendingcr = bool(flag & 1)
        if self.decoder is not None:
            self.decoder.setstate((buf, flag >> 1))

    def reset(self):
        self.seennl = 0
        self.pendingcr = False
        if self.decoder is not None:
            self.decoder.reset()

    @property
    def newlines(self):
        return (
            None,
            "\n",
            "\r",
            ("\r", "\n"),
            "\r\n",
            ("\n", "\r\n"),
            ("\r", "\r\n"),
            ("\r", "\n", "\r\n"),
        )[self.seennl]
