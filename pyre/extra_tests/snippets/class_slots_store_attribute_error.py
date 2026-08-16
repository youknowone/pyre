# pyre-check: gate=1
def make_type():
    class X:
        __slots__ = 'a'
    return X
X = make_type()
try:
    X().b = 1
except AttributeError as exc:
    result = str(exc)

assert result == ("'X' object has no attribute 'b'"
                  " and no __dict__ for setting new attributes")
