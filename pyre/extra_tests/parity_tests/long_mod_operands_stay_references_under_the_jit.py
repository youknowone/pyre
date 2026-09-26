# CPython-suite gap: longobject tests do not run `%`, `//` or divmod on
# nursery-born longs in a loop hot enough to be traced.
# parity-tests reason: the `_divrem` lhs-remainder test in `long_mod` and
# `divmod` must reach its residual as GC references; an address cast to a
# machine word folds into a constant that a minor collection leaves stale.

"""Long `%`, `//` and divmod keep their operands across a minor collection.

Every step allocates fresh longs, so the default nursery collects many times
while the loops run traced. `mix` covers the reflected int receiver and the
remainder that reuses the dividend's payload; `divmod` covers the same
payload reuse on its remainder half.
"""

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

BIGP = 18446744073709551629
BIGA = 100000000000000000000000
ROUNDS = 3000


def mix(h, v):
    return (h * 3 + (v % BIGP)) % BIGP


def floordiv_mix(h, v):
    return (h * 7 + (BIGA + v) // BIGP) % BIGP


def divmod_mix(h, v):
    q, r = divmod(BIGP + v * 5, BIGA)
    return (h * 3 + q + r) % BIGP


# One loop per shape: each traces on its own, the way the defect was found.
h_int = 0
for i in range(ROUNDS):
    h_int = mix(h_int, i)
h_long = 0
for i in range(ROUNDS):
    h_long = mix(h_long, BIGA + i)
h_floordiv = 0
for i in range(ROUNDS):
    h_floordiv = floordiv_mix(h_floordiv, i)
h_divmod = 0
for i in range(ROUNDS):
    h_divmod = divmod_mix(h_divmod, i)

assert h_int == 10169797193228046475, h_int
assert h_long == 13267213028806569560, h_long
assert h_floordiv == 14656875528999295164, h_floordiv
assert h_divmod == 13955497818721129117, h_divmod

print("OK")
