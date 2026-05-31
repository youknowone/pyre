# int/float/str/bool expose __format__(self, format_spec), routing the
# spec through the shared format-spec parser (the same path as format()
# and f-strings).  An empty spec collapses to str(self).  Self-contained
# (no testutils import) so it runs on every backend.

# int
assert (1234).__format__("_d") == "1_234"
assert (255).__format__("#x") == "0xff"
assert (10).__format__("08b") == "00001010"
assert (1234).__format__("") == "1234"
assert (65).__format__("c") == "A"

# float spec on int coerces to float formatting
assert (3).__format__(".2f") == "3.00"

# float
assert (3.14).__format__(".1f") == "3.1"
assert (3.14).__format__("") == "3.14"
assert (1.0).__format__("e") == "1.000000e+00"

# str
assert "hi".__format__(">10") == "        hi"
assert "hi".__format__("") == "hi"
assert "hello".__format__(".3") == "hel"

# bool inherits int.__format__ through the MRO
assert True.__format__("d") == "1"
assert False.__format__("") == "False"

print("builtin_format_dunder: OK")
