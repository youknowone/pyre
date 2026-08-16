# pyre-check: gate=1
# pyopcode.py:1032-1039 — `except <non-exception>:` raises
# TypeError(CANNOT_CATCH_MSG). The bare `except 42:` form is
# syntactically valid; the runtime gate fires in CHECK_EXC_MATCH.
try:
    try:
        raise ValueError("boom")
    except 42:
        pass
except TypeError as exc:
    assert 'BaseException' in str(exc)
else:
    raise AssertionError('`except 42:` must raise TypeError')
