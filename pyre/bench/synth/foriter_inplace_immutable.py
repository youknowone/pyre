# pyre-check: max-pypy-ratio=3.7
# Immutable `+=` at the end of a FOR_ITER body. dynasm and cranelift both
# read 1.8x; the ceiling is twice that.
"""FOR_ITER must not drop an item when immutable ``+=`` ends a hot body."""


class StatefulDecoder:
    def __init__(self) -> None:
        self.buffer = bytearray()

    def process_word(self):
        output = self.buffer.decode("ascii")
        self.buffer = bytearray()
        return output

    def decode(self, data):
        output = ""
        for byte in data:
            self.buffer.append(byte)
            output += self.process_word()
        return output


decoder = StatefulDecoder()
result = decoder.decode(b"abcd" * 60_000)
assert result == "abcd" * 60_000, (len(result), result[:8], result[-8:])
print(len(result), result[:4], result[-4:])
