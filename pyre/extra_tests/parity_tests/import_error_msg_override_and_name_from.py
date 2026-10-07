# CPython-suite gap: `test_exceptions` reads `ImportError.msg`, `name` and
# `path` after construction. It never assigns `msg` afterwards, so `str()`
# following a later override is untested, and it never constructs with
# `name_from=`.
#
# parity-tests reason: `ImportError_str` returns an exact `str` `msg`
# and otherwise lets `BaseException_str` render `args`.
# `ModuleNotFoundError` shares that `tp_str`. `W_ImportError` has no
# `descr_str`, so pypy3 always stringifies `args_w`. `ImportError_init`
# parses `name_from` into the typed slot; `W_ImportError.descr_init` has
# no such keyword.
#
# pyre-check: pypy-diverges: pypy3 answers `str(e) == "m"` after
# `e.msg = "other"` because `W_BaseException.descr_str` ignores `w_msg`,
# and `'name_from' is an invalid keyword argument for ImportError`.
class Msg(str):
    pass


e = ImportError("m")
e.msg = "other"
assert e.msg == "other", e.msg
assert e.args == ("m",), e.args
assert str(e) == "other", str(e)

e = ModuleNotFoundError("m")
e.msg = "other"
assert str(e) == "other", str(e)

for not_exact_str in (Msg("other"), 1, None):
    e = ImportError("m")
    e.msg = not_exact_str
    assert str(e) == "m", (not_exact_str, str(e))

e = ImportError("m", name="n", path="p", name_from="f")
assert e.name == "n", e.name
assert e.path == "p", e.path
assert e.name_from == "f", e.name_from
assert e.args == ("m",), e.args

print("OK")
