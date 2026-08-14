# CPython-suite gap: the exception tests read `StopIteration.value` /
# `NameError.name` / `ImportError.path` once, never from a `raise` site hot
# enough for a JIT construction fold to take over.
# parity-tests reason: pyre builds the zero-argument instance of a bare
# `raise X` in traced code, so those slots are written by trace IR rather than
# by the runtime `__init__`.

# `do_raise` instantiates a raised class with no arguments, so every flattened
# slot the initializer touches takes its no-argument value:
# `interp_exceptions.py:496-499 W_StopIteration` leaves `value` at None,
# `:810-812 W_NameError` and `:1134-1137 W_AttributeError` take `name` / `obj`
# from keywords only, and `:363-377 W_ImportError` fills `name` / `path` /
# `msg` from keywords and a lone positional. A fold that emits those stores
# itself has to reproduce all of them, and leave `args`, `__context__`,
# `__cause__` and `__suppress_context__` exactly as the interpreter does.

N = 3000


def raise_stop_iteration():
    raise StopIteration


def raise_name_error():
    raise NameError


def raise_attribute_error():
    raise AttributeError


def raise_import_error():
    raise ImportError


def raise_module_not_found():
    raise ModuleNotFoundError


def raise_value_error():
    raise ValueError


def caught(fn):
    """The instance a bare `raise` built, over a loop the JIT compiles."""
    seen = None
    for _ in range(N):
        try:
            fn()
        except BaseException as exc:
            seen = exc
            assert exc.args == ()
            assert exc.__cause__ is None
            assert exc.__context__ is None
            assert exc.__suppress_context__ is False
            assert exc.__traceback__ is not None
    return seen


exc = caught(raise_stop_iteration)
assert type(exc) is StopIteration, type(exc)
assert exc.value is None, exc.value

exc = caught(raise_name_error)
assert type(exc) is NameError, type(exc)
assert exc.name is None, exc.name

exc = caught(raise_attribute_error)
assert type(exc) is AttributeError, type(exc)
assert exc.name is None, exc.name
assert exc.obj is None, exc.obj

for fn, cls in ((raise_import_error, ImportError),
                (raise_module_not_found, ModuleNotFoundError)):
    exc = caught(fn)
    assert type(exc) is cls, type(exc)
    assert exc.name is None, exc.name
    assert exc.path is None, exc.path
    assert exc.msg is None, exc.msg

exc = caught(raise_value_error)
assert type(exc) is ValueError, type(exc)

# `raise X from Y` keeps `__cause__` and flips `__suppress_context__`, which the
# no-cause fold must decline rather than reproduce.
cause = ValueError("cause")
for _ in range(N):
    try:
        raise StopIteration from cause
    except StopIteration as exc:
        assert exc.__cause__ is cause
        assert exc.__suppress_context__ is True
        assert exc.value is None

# A raise nested inside an active handler chains `__context__` onto the new
# instance, so the fold's inline `__context__` store has to see the same value.
for _ in range(N):
    try:
        raise ValueError("outer")
    except ValueError as outer:
        try:
            raise StopIteration
        except StopIteration as inner:
            assert inner.__context__ is outer
            assert inner.value is None

print("OK")
