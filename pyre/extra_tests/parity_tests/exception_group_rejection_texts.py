# CPython-suite gap: `test_exception_group` constructs with positionals and
# never `ExceptionGroup(message=..., exceptions=...)`.
# `test_fields_are_readonly` asserts only the AttributeError type of a store
# to `message` / `exceptions`.
#
# parity-tests reason: `BaseExceptionGroup_new` takes two positional
# arguments; keywords are not bound, so the call is `__new__() takes exactly
# 2 arguments (0 given)`. `message` and `exceptions` are readonly members, so
# a store reports `readonly attribute`. `W_BaseExceptionGroup.descr_new`
# binds `w_message` and `w_exceptions` and leaves keyword rejection to
# `descr_init`; `interp_exceptions.py` declares the fields as
# `GetSetProperty(..., fset=None)`, which names the field.
#
# pyre-check: pypy-diverges: pypy3 answers `BaseExceptionGroup.__init__()
# takes no keyword arguments` for the keyword call, and
# `readonly attribute 'message'` / `readonly attribute 'exceptions'` for the
# stores.
def rejects(fn, exc_type, message):
    try:
        fn()
    except exc_type as err:
        assert str(err) == message, err
    else:
        raise AssertionError(f"expected {exc_type.__name__}")


def store(obj, name, value):
    setattr(obj, name, value)


rejects(
    lambda: ExceptionGroup(message="m", exceptions=[ValueError()]),
    TypeError,
    "BaseExceptionGroup.__new__() takes exactly 2 arguments (0 given)",
)

eg = ExceptionGroup("m", [ValueError(1)])
rejects(lambda: store(eg, "message", "x"), AttributeError, "readonly attribute")
rejects(
    lambda: store(eg, "exceptions", (ValueError(1),)),
    AttributeError,
    "readonly attribute",
)

print("OK")
