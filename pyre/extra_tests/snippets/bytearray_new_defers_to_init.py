# pyre-check: gate=1
class Source:
    def __init__(self):
        self.calls = 0
    def __iter__(self):
        self.calls += 1
        return iter((1, 2, 3))

source = Source()
value = bytearray(source)
raw = bytearray.__new__(bytearray, source, ignored=True)

class Sub(bytearray):
    pass

sub_source = Source()
sub = Sub(sub_source)
result = (
    value == bytearray(b'\x01\x02\x03')
    and source.calls == 1
    and raw == bytearray()
    and type(sub) is Sub
    and sub == bytearray(b'\x01\x02\x03')
    and sub_source.calls == 1
)

assert result
