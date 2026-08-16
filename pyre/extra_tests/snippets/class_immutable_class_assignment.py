# pyre-check: gate=1
class MyInt(int):
    __slots__ = ()
try:
    (1).__class__ = MyInt
except TypeError:
    result = True
else:
    result = False

assert result
