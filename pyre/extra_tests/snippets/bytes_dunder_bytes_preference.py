# pyre-check: gate=1
class StringWithBytes(str):
    def __new__(cls, value):
        self = str.__new__(cls, '\u20ac')
        self.value = value
        return self
    def __bytes__(self):
        return self.value

source = StringWithBytes(b'abc')
result = bytes(source) == b'abc'

assert result
