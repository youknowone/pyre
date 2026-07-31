"""Regression coverage for `_functools` argument and nesting semantics."""

from functools import partial, reduce


for args, kwargs in (
    ((), {}),
    ((lambda left, right: left + right,), {}),
    ((), {"initial": [1, 2]}),
    ((lambda left, right: left + right,), {"initial": [1, 2]}),
):
    try:
        reduce(*args, **kwargs)
    except TypeError:
        pass
    else:
        raise AssertionError("a keyword initial filled a positional-only argument")


flat = partial(partial(pow, 2), 5)
assert flat.func is pow
assert flat.args == (2, 5)
assert flat() == 32

print("OK")
