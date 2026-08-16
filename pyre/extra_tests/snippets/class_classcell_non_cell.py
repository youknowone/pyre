# pyre-check: gate=1
rejected = False
try:
    class C:
        __classcell__ = 42
        __slots__ = ['__classcell__']
except TypeError:
    rejected = True

assert rejected
