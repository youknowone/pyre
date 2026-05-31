# A float format spec with no presentation type formats like repr():
# the trailing `.0` of a whole value is kept (grouping, width, sign and
# alignment modifiers must not drop it).  The `g`/`G` types use their own
# default precision of 6.  Self-contained (no testutils import).

# No presentation type keeps the fractional `.0`.
assert format(1000000.0, ",") == "1,000,000.0"
assert format(1234.5, ",") == "1,234.5"
assert format(1000.0, "_") == "1_000.0"
assert format(1.0, "10") == "       1.0"
assert format(1.0, "+") == "+1.0"
assert format(100.0, "<8") == "100.0   "
assert "{:,}".format(1234567.0) == "1,234,567.0"

# repr-style exponential band still applies with no type.
assert format(1e16, "20") == "               1e+16"
assert format(1e-5, ">12") == "       1e-05"

# `g`/`G` with no explicit precision default to precision 6.
assert format(1000000.0, "g") == "1e+06"
assert format(1234.0, "g") == "1234"
assert format(1000000.0, ",g") == "1e+06"

# Fixed/exponential types are unaffected.
assert format(1000000.0, ",.2f") == "1,000,000.00"
assert format(1234.5, ".1f") == "1234.5"

print("format_float_no_type: OK")
