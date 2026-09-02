# CPython-suite gap: the suite never sets __defaults__ to a tuple longer than
# the function's positional parameter count, so nothing there distinguishes
# aligning the defaults at the tuple's head from aligning them at its tail.
# parity-tests reason: the divergence is silent — the call returns a value,
# just the wrong element of __defaults__ — so only a value check catches it.

"""__defaults__ longer than the positional parameters binds from its tail.

`argument.py` `_match_signature` computes `def_first = co_argcount -
len(defaults_w)` as a *signed* quantity and then reads `defaults_w[i -
def_first]`.  When `__defaults__` is longer than the positional parameters
`def_first` is negative, which shifts every read towards the end of the
tuple.  Clamping it at zero instead reads the head and silently binds the
wrong value.

`__defaults__` is writable, so the over-long tuple is reachable from plain
Python without any introspection trickery.
"""


def two(a, b):
    return a, b


two.__defaults__ = (100, 101, 102)

# The positional-only binder and the keyword binder must agree about the same
# call: both take defaults[1] and defaults[2], never defaults[0].
assert two() == (101, 102)
assert two(a=9) == (9, 102)
assert two(b=9) == (101, 9)
assert two(9) == (9, 102)

two.__defaults__ = (100, 101, 102, 103, 104)
assert two() == (103, 104)
assert two(a=9) == (9, 104)
assert two(b=9) == (103, 9)

# One default per parameter is the aligned case, and one fewer still binds the
# tail — the same rule, with def_first non-negative.
two.__defaults__ = (100, 101)
assert two(a=9) == (9, 101)
two.__defaults__ = (100,)
assert two(a=9) == (9, 100)


def three(a, b, c):
    return a, b, c


three.__defaults__ = (100, 101, 102, 103, 104)
assert three() == (102, 103, 104)
assert three(a=9) == (9, 103, 104)
assert three(c=9) == (102, 103, 9)

# Keyword-only defaults come from __kwdefaults__ by name and must be untouched
# by the positional alignment.
def kwonly(a, b, *, k=None):
    return a, b, k


kwonly.__defaults__ = (100, 101, 102)
kwonly.__kwdefaults__ = {"k": 200}
assert kwonly() == (101, 102, 200)
assert kwonly(k=9) == (101, 102, 9)

# `ArgErrTooMany.getmsg` formats `num_args - num_defaults` and that difference
# is negative here.  It has to be reported as a negative number, not wrapped
# through an unsigned width.
def varkw(a, **kw):
    return a, kw


varkw.__defaults__ = (100, 101)
try:
    varkw(1, 2, 3, x=4)
except TypeError as exc:
    assert "takes from -1 to 1 positional arguments" in str(exc), exc
else:
    raise AssertionError("three positionals into a one-parameter function bound")

varkw.__defaults__ = (100, 101, 102)
try:
    varkw(1, 2, 3, x=4)
except TypeError as exc:
    assert "takes from -2 to 1 positional arguments" in str(exc), exc
else:
    raise AssertionError("three positionals into a one-parameter function bound")

# The same call compiled: the binder has a second copy on the traced path, and
# a loop long enough to compile reaches it.
def warm():
    for _ in range(50000):
        assert two(a=9) == (9, 100)
        assert three(c=9) == (102, 103, 9)


two.__defaults__ = (100,)
warm()

print("OK")
