# PEP 682 `z` format option: coerce a negative-zero result to positive zero.
# The float renderings below are byte-identical across CPython and PyPy; the
# integer `z` rejection differs only in message wording, so this fixture checks
# its exception type instead.

float_cases = [
    ("z.2f", -0.0),
    ("z.2f", -0.04),
    ("z.1f", -0.04),
    ("z.2f", -1.5),
    ("z.2f", 0.0),
    ("z.2f", 1.5),
    ("z.3e", -0.0004),
    ("z.1e", -0.04),
    ("z.2e", -0.0),
    ("z.2g", -0.0004),
    ("+z.2f", -0.0),
    (" z.2f", -0.0),
    ("z10.2f", -0.0004),
    ("z.0f", -0.4),
    ("z%", -0.0),
    ("z", -0.0),
    ("z.2f", -1e-10),
]
for spec, val in float_cases:
    print(f"{spec!r:8} {val!r:8} -> {format(val, spec)!r}")

# `z` alone requires a float presentation type; an integer rejects it.
try:
    format(5, "z")
    print("int z: no error")
except ValueError as e:
    print(f"int z: ValueError (msg-len {len(str(e)) > 0})")

# A float presentation code on an int keeps working with `z` present.
print("int z.2f ->", format(5, "z.2f"))
