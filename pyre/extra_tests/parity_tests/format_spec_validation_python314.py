def check(value, spec, expected):
    try:
        format(value, spec)
    except ValueError as exc:
        assert str(exc) == expected, (value, spec, str(exc))
    else:
        raise AssertionError((value, spec, "ValueError not raised"))


for value in (12, 12j):
    check(
        value,
        "%M",
        f"Invalid format specifier '%M' for object of type '{type(value).__name__}'",
    )

check(
    "x",
    "zs",
    "Negative zero coercion (z) not allowed in string format specifier",
)

for spec in (".,_f", "._,f"):
    check(1.1, spec, "Cannot specify both ',' and '_'.")

for value in (1, 1.1, 1j, "x"):
    for spec, code in ((".,,", ","), (".__", "_")):
        check(
            value,
            spec,
            f"Unknown format code '{code}' for object of type "
            f"'{type(value).__name__}'",
        )

# Integral and fractional grouping deliberately may use different separators;
# only two alternatives in the same slot conflict.
assert format(1234.5, ",._f") == "1,234.500_000"
assert format(1234.5, "_.,f") == "1_234.500,000"

print("OK")
