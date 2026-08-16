# pyre-check: gate=1
# pyopcode.py:1034-1037 — tuple form, any non-exception entry
# raises TypeError. `except (ValueError, 42):` must trigger the
# gate even though `ValueError` itself is valid.
try:
    try:
        raise ValueError("boom")
    except (ValueError, 42):
        pass
except TypeError as exc:
    assert 'BaseException' in str(exc)
else:
    raise AssertionError('`except (ValueError, 42):` must raise TypeError')
