"""`code.co_exceptiontable` is exposed as a `bytes` attribute.

`pypy/interpreter/typedef.py:720` declares
`co_exceptiontable = interp_attrproperty('co_exceptiontable', cls=PyCode,
                                          wrapfn="newbytes")`.
A function with any try/except must have a non-empty exception table;
a function with no exception handling has an empty one.
"""


def with_try(x):
    try:
        return int(x)
    except ValueError:
        return -1


def without_try(x):
    return x + 1


assert isinstance(with_try.__code__.co_exceptiontable, bytes), (
    type(with_try.__code__.co_exceptiontable),
)
assert len(with_try.__code__.co_exceptiontable) > 0, "try/except function should have entries"
assert isinstance(without_try.__code__.co_exceptiontable, bytes), (
    type(without_try.__code__.co_exceptiontable),
)
assert len(without_try.__code__.co_exceptiontable) == 0, (
    "no-except function should have empty table"
)

print("OK")
