# pyre-check: gate=1
try:
    raise int
except TypeError as exc:
    assert str(exc) == 'exceptions must derive from BaseException'
else:
    raise AssertionError('`raise int` must be a TypeError')
