# pyre-check: gate=1
format_string = bytearray(b'%a end')

class Value:
    def __repr__(self):
        format_string.clear()
        return 'value'

try:
    format_string % Value()
except BufferError:
    preserved = format_string == bytearray(b'%a end')
    format_string.clear()
    released = format_string == bytearray()
    result = preserved and released
else:
    result = False

assert result
