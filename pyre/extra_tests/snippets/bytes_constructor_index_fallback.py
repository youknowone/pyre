# pyre-check: gate=1
class BufferWithIndex(bytes):
    def __index__(self):
        raise TypeError

class IterableWithIndex:
    def __index__(self):
        raise TypeError
    def __iter__(self):
        return iter((102, 111, 111))

buffer = BufferWithIndex(b'foobar')
result = (
    bytes(buffer) == b'foobar'
    and bytearray(buffer) == bytearray(b'foobar')
    and bytes(IterableWithIndex()) == b'foo'
    and bytearray(IterableWithIndex()) == bytearray(b'foo')
)

assert result
