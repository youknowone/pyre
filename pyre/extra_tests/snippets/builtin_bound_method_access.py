# pyre-check: gate=1
class C:
    def add(self, x):
        return x + 1
c = C()
m = c.add
result = m(41)

assert result == 42
