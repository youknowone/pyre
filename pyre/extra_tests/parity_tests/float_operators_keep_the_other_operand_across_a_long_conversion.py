# CPython-suite gap: test_float.test_floatasratio divides a fresh float by a
# long only 10000 times, which crashes a few runs in ten.
# parity-tests reason: converting a long operand to a double allocates
# (`rbigint.tofloat`), so the float operator must read its other, nursery-born
# operand back after the conversion; a stale one reads its forwarding word.

"""Mixed float/long and complex/long operators under a moving collection.

Every iteration boxes fresh floats and complexes, so the nursery collects
many times while a long operand is being converted to a double.
"""

d = 2**250 + 12345
fd = float(d)
c = complex(1.5, 2.0)

for i in range(60000):
    x = 1.5 + (i & 7)
    assert x.__truediv__(d) == x / fd
    assert x / d == x / fd
    assert d / x == fd / x
    assert x + d == x + fd
    assert d - x == fd - x
    assert x * d == x * fd
    assert d // x == fd // x
    assert d % x == fd % x
    assert divmod(d, x) == divmod(fd, x)
    assert x ** -d == x ** -fd
    assert d ** -x == fd ** -x
    assert pow(x, 2) == x * x
    z = c + x
    assert z * d == z * fd
    assert d / z == fd / z
    assert z + d == z + fd

try:
    1.5 / (2**2000)
except OverflowError as e:
    assert str(e) == "int too large to convert to float", e
else:
    raise AssertionError("an over-range long operand must raise")

print("OK")
