class BytesBuilder(object):
    """Append-only byte buffer with O(1) amortized growth.

    Accumulates byte chunks and yields the concatenation via build().
    Matches the surface pickle.py uses: append(data), build(), len().
    """

    def __init__(self, initial=0):
        self._buf = bytearray()

    def append(self, data):
        self._buf += data

    def build(self):
        return bytes(self._buf)

    def __len__(self):
        return len(self._buf)


class StringBuilder(object):
    """Append-only text buffer; build() returns the joined str."""

    def __init__(self, initial=0):
        self._parts = []
        self._len = 0

    def append(self, data):
        self._parts.append(data)
        self._len += len(data)

    def build(self):
        return ''.join(self._parts)

    def __len__(self):
        return self._len
