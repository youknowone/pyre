# pyre-check: gate=1
class C:
    def add(self, x):
        return x + 1
c = C()
m = c.add

assert type(m).__name__ == 'method'
assert m.__self__ is c
