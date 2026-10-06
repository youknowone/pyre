# CPython-suite gap: `test_exceptions` reads `ImportError.msg` after
# construction and never assigns `msg` afterwards, so `str()` following
# a later override is untested.
#
# parity-tests reason: `ImportError_str` returns an exact `str` `msg`
# and otherwise lets `BaseException_str` render `args`.
# `ModuleNotFoundError` shares that `tp_str`. `W_ImportError` has no
# `descr_str`, so pypy3 always stringifies `args_w`.
#
# pyre-check: pypy-diverges: pypy3 answers `str(e) == "m"` after
# `e.msg = "other"` because `W_BaseException.descr_str` ignores `w_msg`.
e = ImportError("m")
e.msg = "other"
assert e.msg == "other", e.msg
assert e.args == ("m",), e.args
assert str(e) == "other", str(e)

e = ModuleNotFoundError("m")
e.msg = "other"
assert str(e) == "other", str(e)

e = ImportError("m")


class Msg(str):
    pass


e.msg = Msg("other")
assert str(e) == "m", str(e)

e = ImportError("m")
e.msg = 1
assert str(e) == "m", str(e)

e = ImportError("m")
e.msg = None
assert str(e) == "m", str(e)

print("OK")
