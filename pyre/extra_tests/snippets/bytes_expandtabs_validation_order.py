# pyre-check: gate=1
result = True
for value in (b'a\tb', bytearray(b'a\tb')):
    for call, expected in (
        (lambda: value.expandtabs(2, 3),
         'expandtabs() takes at most 1 argument (2 given)'),
        (lambda: value.expandtabs(one=1, two=2),
         'expandtabs() takes at most 1 keyword argument (2 given)'),
    ):
        try:
            call()
        except TypeError as exc:
            result = result and str(exc) == expected
        else:
            result = False

value = bytearray(b'a\tb')
class TabSize:
    def __index__(self):
        value[:] = b'x\ty'
        return 4
result = result and value.expandtabs(TabSize()) == bytearray(b'x   y')

# CPython 3.14's clinic wrapper converts `tabsize` with `PyLong_AsInt`
# before any receiver fast path.  This applies uniformly to str, bytes and
# bytearray, including an empty value that needs no expansion.
for value in ('', b'', bytearray()):
    for tabsize in (2 ** 31, -(2 ** 31) - 1, 2 ** 100):
        try:
            value.expandtabs(tabsize)
        except OverflowError as exc:
            result = result and str(exc) == (
                'Python int too large to convert to C int'
            )
        else:
            result = False

assert result
