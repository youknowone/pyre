# pyre-check: gate=1
exc = ValueError()
cause = KeyError()
try:
    raise exc from cause
except ValueError as caught:
    assert caught is exc
    assert exc.__cause__ is cause
else:
    raise AssertionError('`raise ... from ...` must propagate the exception')
