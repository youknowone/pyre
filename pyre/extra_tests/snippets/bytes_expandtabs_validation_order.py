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

assert result
