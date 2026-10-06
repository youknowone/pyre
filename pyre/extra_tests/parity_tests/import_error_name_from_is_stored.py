# CPython-suite gap: `test_exceptions` reads ImportError.name and
# ImportError.path and never constructs with `name_from=`.
#
# parity-tests reason: `ImportError_init` parses `name_from` into the
# typed slot. `W_ImportError.descr_init` has no such keyword, so pypy3
# rejects it as invalid.
#
# pyre-check: pypy-diverges: pypy3 answers
# `'name_from' is an invalid keyword argument for ImportError`.
e = ImportError("m", name="n", path="p", name_from="f")
assert e.name == "n", e.name
assert e.path == "p", e.path
assert e.name_from == "f", e.name_from
assert e.args == ("m",), e.args

print("OK")
