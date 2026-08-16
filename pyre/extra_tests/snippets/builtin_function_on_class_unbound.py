# pyre-check: gate=1
class C:
    f = len
c = C()
result = c.f([1, 2, 3])

assert result == 3
