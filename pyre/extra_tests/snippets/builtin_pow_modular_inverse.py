# 3-arg pow() with a negative exponent computes the modular inverse of
# `base` raised to `-exp` (Python 3.8+).  Self-contained (no testutils
# import) so it runs on every backend.

# Basic modular inverse: pow(b, -1, m) is the inverse of b modulo m.
assert pow(5, -1, 13) == 8
assert (5 * 8) % 13 == 1

assert pow(3, -1, 7) == 5
assert (3 * 5) % 7 == 1

# Negative exponents other than -1.
assert pow(3, -3, 7) == 6
assert pow(2, -5, 9) == pow(pow(2, -1, 9), 5, 9)

# A base that shares a factor with the modulus has no inverse.
raised = False
try:
    pow(2, -1, 4)
except ValueError:
    raised = True
assert raised, "non-invertible base should raise ValueError"

# Negative modulus mirrors the sign convention of 3-arg pow.
assert pow(3, -1, -7) == -2
assert pow(5, -1, -13) == -5

# Modulus of magnitude 1 always yields 0.
assert pow(3, -1, 1) == 0
assert pow(3, -1, -1) == 0

print("builtin_pow_modular_inverse: OK")
