# CPython-suite gap: `test_exceptions` constructs UnicodeError subclasses
# with five (or four) positionals and never keyword-only field names.
# `ValueError(foo=1)` is also absent from the suite, as are keyword
# calls on MemoryError, KeyError, IndentationError, Warning, and OSError
# subclasses.
#
# parity-tests reason: `_PyArg_NoKeywords` inside `BaseException_init`
# and the UnicodeError inits reports `{Type}() takes no keyword
# arguments`, using the receiver type. `W_UnicodeDecodeError.descr_init`
# / `W_UnicodeEncodeError.descr_init` bind the five fields as interp2app
# arguments, so pypy3 accepts those keywords. `W_BaseException.descr_init`
# names `BaseException.__init__`. `W_SyntaxError.descr_init` names
# `SyntaxError.__init__` for IndentationError and TabError.
# `W_OSError.descr_new` says `OSError does not take keyword arguments`
# for every OSError subclass. `W_SystemExit.descr_init` and
# `W_StopIteration.descr_init` name their own `.__init__`.
#
# pyre-check: pypy-diverges: pypy3 constructs UnicodeDecodeError and
# UnicodeEncodeError from keyword field names successfully.
# UnicodeError, ValueError, MemoryError, and KeyError keyword calls
# raise `BaseException.__init__() takes no keyword arguments`.
# UnicodeTranslateError names `UnicodeTranslateError.__init__`.
# IndentationError and TabError name `SyntaxError.__init__`.
# BlockingIOError and FileNotFoundError raise
# `OSError does not take keyword arguments`. A MemoryError subclass
# still names `BaseException.__init__`. SystemExit and StopIteration
# name `SystemExit.__init__` / `StopIteration.__init__`.
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
reject(lambda: MemoryError(foo=1), "MemoryError() takes no keyword arguments")
reject(lambda: KeyError(foo=1), "KeyError() takes no keyword arguments")
reject(lambda: KeyboardInterrupt(foo=1), "KeyboardInterrupt() takes no keyword arguments")
reject(
    lambda: IndentationError(foo=1),
    "IndentationError() takes no keyword arguments",
)
reject(lambda: TabError(foo=1), "TabError() takes no keyword arguments")
reject(
    lambda: BlockingIOError(foo=1),
    "BlockingIOError() takes no keyword arguments",
)
reject(
    lambda: FileNotFoundError(foo=1),
    "FileNotFoundError() takes no keyword arguments",
)


class ME(MemoryError):
    pass


reject(lambda: ME(foo=1), "ME() takes no keyword arguments")

# One representative per constructor path: the shared `BaseException_init`,
# the types with their own init, and OSError's `__new__`/`__init__` split.
for T in (
    BaseException,
    Exception,
    GeneratorExit,
    SystemExit,
    StopIteration,
    Warning,
    SyntaxError,
    OSError,
    PermissionError,
):
    reject(lambda T=T: T(foo=1), f"{T.__name__}() takes no keyword arguments")


class PE(PermissionError):
    pass


reject(lambda: PE(foo=1), "PE() takes no keyword arguments")

print("OK")
