# pyre-check: max-pypy-ratio=34
# pypy's exec time is pinned to the startup-subtraction floor on some hosts,
# so the ratio is not a measurement everywhere: the ceiling is twice the
# slowest the CI runners observe (16.9x), rounded up. It read 36 before a later
# tightening fitted it to a single run's numbers.
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
result = decoder.decode(b"abcd" * 2_000)
assert result == "abcd" * 2_000, (len(result), result[:8], result[-8:])
print(len(result), result[:4], result[-4:])
