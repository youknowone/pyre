# CPython-suite gap: `test_zipfile` does own this one -- `ZipExtFile.read` is
# the shape it was found in -- but it owns it as a flake.  The wrong answer
# needs a guard failure to land on a particular already-compiled trace, so the
# module failed about one run in seven and passed the rest, and a module that
# passes six times out of seven reads as green.  Nothing in the suite pins the
# arithmetic that makes it decidable in one run.
#
# parity-tests reason: a buffered reader hands out the tail of its buffer and
# then empties it, and the two are one indivisible step: whatever `read` slices
# out of the buffer before clearing it has to reach the caller.  Every
# self-consistent reader gives the same answer here, so the byte count is a
# whole-language invariant rather than a detail of this one class -- and
# `n > want` below is unreachable in any of them, because the amount still owed
# after the buffer is drained cannot exceed what was asked for.  A runtime that
# runs the region twice, or reads a field it has already written, breaks both at
# once: it returns a short stream AND reports an impossible `n`.
BAD = []


class Reader:
    """A buffered reader over `data`, refilled in `sizes`-shaped chunks."""

    def __init__(self, data, sizes):
        self._data = data
        self._sizes = sizes
        self._pos = 0
        self._si = 0
        self._readbuffer = b""
        self._offset = 0
        self._eof = False

    def _next_chunk(self, n):
        if self._eof or n <= 0:
            return b""
        take = self._sizes[self._si % len(self._sizes)]
        self._si += 1
        chunk = self._data[self._pos : self._pos + take]
        self._pos += len(chunk)
        if self._pos >= len(self._data):
            self._eof = True
        return chunk

    def read(self, n):
        want = n
        end = n + self._offset
        if end < len(self._readbuffer):
            buf = self._readbuffer[self._offset : end]
            self._offset = end
            return buf
        n = end - len(self._readbuffer)
        # Reads no field: `n` is what is still owed once the buffer is drained,
        # so it is bounded by `want` on any reader that agrees with itself about
        # how much the buffer held.  Asking through the fields instead would
        # answer whatever the fields say NOW, which is the thing in question.
        if n > want:
            BAD.append((want, n))
        buf = self._readbuffer[self._offset :]
        self._readbuffer = b""
        self._offset = 0
        while n > 0 and not self._eof:
            data = self._next_chunk(n)
            if n < len(data):
                self._readbuffer = data
                self._offset = n
                buf += data[:n]
                break
            buf += data
            n -= len(data)
        return buf


def drain(data, sizes):
    r = Reader(data, sizes)
    out = []
    while True:
        b = r.read(256)
        if not b:
            break
        out.append(b)
    return b"".join(out)


src = bytes(range(256)) * 1024
# Two shapes, in this order: the first is what gets compiled, and the second
# enters that compiled code with a chunk size it never traced, so the deopt
# lands mid-`read` rather than at a boundary.
drain(src, (97, 1))
BAD.clear()
got = drain(src, (1024, 1))

assert not BAD, f"read() was owed more than it was asked for: {BAD}"
assert len(got) == len(src), f"stream is short by {len(src) - len(got)} bytes"
assert got == src, "stream is the right length but not the right bytes"
print("OK")
