# pyre-check: gate=1
x = 1
class C:
    y = 2
    snap = locals()
result = C.snap['y'] + globals()['x']

assert result == 3
