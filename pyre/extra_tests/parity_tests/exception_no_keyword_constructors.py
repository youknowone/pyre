# CPython-suite gap: `test_exceptions` constructs UnicodeError subclasses
# with five (or four) positionals and never keyword-only field names.
# `ValueError(foo=1)` is also absent from the suite.
#
# parity-tests reason: `_PyArg_NoKeywords` inside `BaseException_init` and
# the UnicodeError inits reports `{Type}() takes no keyword arguments`.
# `W_UnicodeDecodeError.descr_init` / `W_UnicodeEncodeError.descr_init`
# bind the five fields as interp2app arguments, so pypy3 accepts the
# keywords. `W_BaseException.descr_init` names `BaseException.__init__`.
#
# pyre-check: pypy-diverges: pypy3 constructs UnicodeDecodeError and
# UnicodeEncodeError from keyword field names successfully.
# UnicodeError and ValueError keyword calls raise
# `BaseException.__init__() takes no keyword arguments`.
# UnicodeTranslateError names `UnicodeTranslateError.__init__`.
def reject(fn, message):
    try:
        fn()
    except TypeError as err:
        assert str(err) == message, err
    else:
        raise AssertionError("expected TypeError")


reject(
    lambda: UnicodeDecodeError(encoding="u", object=b"x", start=0, end=1, reason="r"),
    "UnicodeDecodeError() takes no keyword arguments",
)
reject(
    lambda: UnicodeEncodeError(encoding="u", object="x", start=0, end=1, reason="r"),
    "UnicodeEncodeError() takes no keyword arguments",
)
reject(
    lambda: UnicodeTranslateError(object="x", start=0, end=1, reason="r"),
    "UnicodeTranslateError() takes no keyword arguments",
)
reject(lambda: UnicodeError(encoding="u"), "UnicodeError() takes no keyword arguments")
reject(lambda: ValueError(foo=1), "ValueError() takes no keyword arguments")
reject(lambda: ValueError().__init__(foo=1), "ValueError() takes no keyword arguments")

print("OK")
