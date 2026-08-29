# CPython-suite gap: zipfile's chunked-read tests assert only the total of a
# whole-member read, so a call that returns more than it was asked for and a
# later one that returns less cancel out and never reach an assertion.
# parity-tests reason: the loss is a pyre bridge-drain replay defect that any
# buffered reader with this refill shape reaches, not a zipfile one.
# parity-env: PYRE_JIT=threshold=40,function_threshold=40

"""A refill that clears the buffer must not lose the bytes already in it.

`read` hands back the buffered tail, clears `_readbuffer`, and only then
enters the loop that refills it -- so a re-entry anywhere between the two
recomputes the request size and the tail from a buffer that is already
empty, asks `_read1` for the whole window instead of the remainder, and
drops the tail it had just taken.  The loss is invisible per call: the
short read is a LONGER return, and only the total says bytes went missing.

The two kinds differ in whether a refill lands on a multiple of the read
size, which decides whether the slow path leaves an empty buffer behind
for the next call; the second kind's varying refill is what puts a live
tail in front of the clear.
"""

CHUNK = 4096
STEP = 256
TOTAL = 600000


class Reader:
    def __init__(self, jitter, total):
        self._jitter = jitter
        self._readbuffer = b""
        self._offset = 0
        self._eof = False
        self._left = total
        self._state = 7

    def _fill(self, n):
        if n < CHUNK:
            n = CHUNK
        if self._jitter:
            self._state = (self._state * 1103515245 + 12345) & 0x7FFFFFFF
            n += 300 + (self._state >> 16) % 560
        if n > self._left:
            n = self._left
        return b"\x01" * n

    def _read1(self, n):
        if self._eof or n <= 0:
            return b""
        data = self._fill(n)
        self._left -= len(data)
        if self._left <= 0:
            self._eof = True
        return data

    def read(self, n):
        end = n + self._offset
        if end < len(self._readbuffer):
            buf = self._readbuffer[self._offset:end]
            self._offset = end
            return buf
        n = end - len(self._readbuffer)
        buf = self._readbuffer[self._offset:]
        self._readbuffer = b""
        self._offset = 0
        while n > 0 and not self._eof:
            data = self._read1(n)
            if n < len(data):
                self._readbuffer = data
                self._offset = n
                buf += data[:n]
                break
            buf += data
            n -= len(data)
        return buf


def drain(jitter):
    reader = Reader(jitter, TOTAL)
    got = 0
    oversized = 0
    while True:
        chunk = reader.read(STEP)
        if not chunk:
            break
        if len(chunk) > STEP:
            oversized += 1
        got += len(chunk)
    return got, oversized


for _ in range(4):
    assert drain(0) == (TOTAL, 0)
for _ in range(4):
    assert drain(1) == (TOTAL, 0)

print("OK")
