# CPython-suite gap: tuple tests omit 3.14 identity across PyPy float specialization.
# parity-tests reason: this targets PyPy/pyre specialised tuple representation.

# A tuple retains the element objects it was built from rather than
# reconstructing them on read.  `is` between two plain floats answers by value
# (`objspace/std/objspace.py:466 is_w` compares `W_FloatObject.floatval`), so
# two equal floats are never distinguishable here; use different values for the
# pair that has to stay distinguishable.

x = float("0.125")
y = float("0.25")
assert x is not y

same = (x, x)
assert same[0] is x
assert same[1] is x
assert same[0] is same[1]

distinct = (x, y)
assert distinct[0] is x
assert distinct[1] is y
assert distinct[0] is not distinct[1]
print("OK")
