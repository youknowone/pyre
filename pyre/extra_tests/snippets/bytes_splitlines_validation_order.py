# pyre-check: gate=1
result = True
for value in (b'a\nb', bytearray(b'a\nb')):
    for call, expected in (
        (lambda: value.splitlines(1, 2),
         'splitlines() takes at most 1 argument (2 given)'),
        (lambda: value.splitlines(one=1, two=2),
         'splitlines() takes at most 1 keyword argument (2 given)'),
    ):
        try:
            call()
        except TypeError as exc:
            result = result and str(exc) == expected
        else:
            result = False

value = bytearray(b'a\nb')
class KeepEnds:
    def __bool__(self):
        value[:] = b'x\ny'
        return True
result = result and value.splitlines(KeepEnds()) == [
    bytearray(b'x\n'), bytearray(b'y')
]

assert result
