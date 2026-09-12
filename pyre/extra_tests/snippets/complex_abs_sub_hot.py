# pyre-check: gate=1
"""`abs(x - y)` over complexes, long enough to compile.

Walking `binary_value_from_tag` for a complex subtract recorded
`w_complex_new` as a bare `object` allocation. After the loop compiled,
`abs` saw `<object object at ...>` and raised TypeError. The helper must
be residualized so the real `complex_sub` builds a `W_ComplexObject`.
"""

xs = [complex(float(i), float(j)) for i in range(-5, 6) for j in range(-5, 6)]
n = 0
for x in xs:
    for y in xs:
        d = x - y
        a = abs(d)
        assert type(d) is complex, (n, type(d), d)
        assert type(a) is float, (n, type(a), a)
        n += 1
assert n == 11 * 11 * 11 * 11, n
