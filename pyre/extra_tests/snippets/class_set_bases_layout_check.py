# pyre-check: gate=1
class O:
    pass
class X:
    pass
rejected = False
try:
    X.__bases__ = (O, type(None))
except TypeError:
    rejected = True

assert rejected
