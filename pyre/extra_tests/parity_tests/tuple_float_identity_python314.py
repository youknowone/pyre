# Python 3.14 tuples retain their input object references. PyPy's unboxed
# float-pair optimisation is observable in pyre because `is` uses pointers.

x = float("0.125")
y = float("0.125")
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
