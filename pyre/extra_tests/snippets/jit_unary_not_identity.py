# pyre-check: gate=1
"""Compiled `not` must keep the bool singletons and the int/bool mix.

The interpreter UNARY_NOT is `space.not_` (`newbool(not is_true(x))`).
The JIT still residualizes that helper (walking `w_bool_from` would
guard one singleton). After warmup the residual fold has to answer
the same objects the interpreter does.
"""

n = 0
for i in range(4000):
    assert (not i) is (i == 0), i
    assert (not (i & 1)) is ((i & 1) == 0), i
    flag = bool(i & 1)
    assert (not flag) is (not bool(i & 1)), i
    n += 1
assert n == 4000
assert (not 0) is True
assert (not 1) is False
assert (not False) is True
assert (not True) is False
assert (not 0.0) is True
assert (not 2.5) is False
assert (not "") is True
assert (not "x") is False
assert (not []) is True
assert (not [1]) is False
