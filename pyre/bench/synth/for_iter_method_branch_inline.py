# pyre-check: max-pypy-ratio=20
# pyre-check: min-pypy-ratio=0.1
# A FOR_ITER caller should inline a method-form callee whose body branches on a
# field.  The callee body carries a `truth` residual before trace-time folding;
# the replay-safety scan must defer it so the walker can erase it instead of
# compiling a separate function-entry trace for the method.
N = 120000


class Cursor:
    def __init__(self):
        self._pos = 11
        self._closed = False

    def m(self, i):
        if self._closed:
            return i
        return self._pos + i


c = Cursor()
total = 0
for i in range(N):
    total = total + c.m(i & 31)
print(total)
