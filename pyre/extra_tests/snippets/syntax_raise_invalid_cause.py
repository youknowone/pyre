# pyre-check: gate=1
try:
    raise ValueError() from 1
except TypeError as exc:
    assert str(exc) == 'exception causes must derive from BaseException'
else:
    raise AssertionError('a non-exception cause must be a TypeError')
