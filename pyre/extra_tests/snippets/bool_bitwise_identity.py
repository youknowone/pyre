# pyre-check: gate=1
"""Compiled `True & True` must stay the bool singleton, not int 1.

`try_emit_exact_int_binop` unboxes bools as ints. The bitwise result
has to go back through `w_bool_from`, or a hot `flag & mask` becomes
a `W_IntObject` and `is True` fails.
"""

n = 0
for _ in range(4000):
    v = True & True
    assert v is True, (n, type(v), v)
    v = True | False
    assert v is True, (n, type(v), v)
    v = True ^ True
    assert v is False, (n, type(v), v)
    n += 1
assert n == 4000
