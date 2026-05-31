# The `n` presentation type formats a float like `g` (its digit grouping
# comes from the locale, which is empty in the C locale) and an int like
# `d`.  Self-contained (no testutils import).

# Float `n` == `g`: default precision 6, general formatting.
assert format(1000000.0, "n") == "1e+06"
assert format(1e16, "n") == "1e+16"
assert format(1234.5, ".2n") == "1.2e+03"
assert format(12345.678, ".4n") == "1.235e+04"
assert format(1234.5, "n") == "1234.5"
assert "{:n}".format(3.14) == "3.14"

# Int `n` == decimal (no locale grouping in the C locale).
assert format(1000000, "n") == "1000000"
assert format(1234, "n") == "1234"
assert format(-5, "n") == "-5"
assert format(0, "n") == "0"
assert format(1234, "10n") == "      1234"

print("format_n_type: OK")
