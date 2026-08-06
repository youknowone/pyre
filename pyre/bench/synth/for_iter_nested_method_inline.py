# pyre-check: max-pypy-ratio=20
# A FOR_ITER caller should inline a method-form callee whose body performs a
# nested method-form call.  The nested `LOAD_METHOD` path emits a
# `load_method_self` residual after the attribute lookup; deferring it lets the
# record-time fold remove it before the replay backstop.
N = 120000


class Cursor:
    def __init__(self):
        self._pos = 11

    def check(self):
        return None

    def m(self, i):
        self.check()
        return self._pos + i


c = Cursor()
total = 0
for i in range(N):
    total = total + c.m(i & 31)
print(total)
