# CPython-suite gap: the exception tests build `AttributeError(msg)` /
# `NameError(msg)` / `StopIteration()` once each, never from a call site hot
# enough for a JIT construction fold to take over.
# parity-tests reason: pyre folds the called form `Type(args)` into trace IR
# once the argument list leaves every flattened slot defaulted, so those slots
# are written by emitted stores rather than by the runtime `__init__`.

# The bare-class sibling (`raise_bare_class_slot_defaults.py`) censuses an
# instance built with NO arguments, so every slot it reads is a trace-time
# constant.  The called form's census reads an instance built from runtime
# operands, and only the callable is guarded — a slot that reads `None` merely
# because THIS iteration's argument was `None` must not be emitted as a
# constant.  `raise StopIteration(x)` covers exactly that: it is traced first
# with `x = None` and then run with a real value.

N = 3000


def caught(fn, exc_type):
    try:
        fn()
    except exc_type as exc:
        return exc
    raise AssertionError("expected a raise")


def raise_attribute_error():
    raise AttributeError("attr message")


def raise_name_error():
    raise NameError("name message")


def raise_stop_iteration():
    raise StopIteration


def raise_import_error():
    raise ImportError


def raise_import_error_pair():
    raise ImportError("import message", "second")


def raise_value_error():
    raise ValueError("value message")


# `interp_exceptions.py:1134-1137 W_AttributeError` takes `name` / `obj` from
# keywords only, so a lone positional leaves both unset.
for _ in range(N):
    exc = caught(raise_attribute_error, AttributeError)
assert type(exc) is AttributeError, type(exc)
assert exc.args == ("attr message",), exc.args
assert exc.name is None, exc.name
assert exc.obj is None, exc.obj

# `:810-812 W_NameError` likewise takes `name` from a keyword only.
for _ in range(N):
    exc = caught(raise_name_error, NameError)
assert type(exc) is NameError, type(exc)
assert exc.args == ("name message",), exc.args
assert exc.name is None, exc.name

# `:496-499 W_StopIteration` defaults `value` to None at zero arity.
for _ in range(N):
    exc = caught(raise_stop_iteration, StopIteration)
assert type(exc) is StopIteration, type(exc)
assert exc.args == (), exc.args
assert exc.value is None, exc.value

# `:363-377 W_ImportError` fills `msg` from a lone positional and leaves it
# unset at every other arity, so zero and two arguments default it.
for _ in range(N):
    exc = caught(raise_import_error, ImportError)
assert type(exc) is ImportError, type(exc)
assert exc.args == (), exc.args
assert exc.name is None, exc.name
assert exc.path is None, exc.path
assert exc.msg is None, exc.msg

for _ in range(N):
    exc = caught(raise_import_error_pair, ImportError)
assert exc.args == ("import message", "second"), exc.args
assert exc.name is None, exc.name
assert exc.path is None, exc.path
assert exc.msg is None, exc.msg

for _ in range(N):
    exc = caught(raise_value_error, ValueError)
assert type(exc) is ValueError, type(exc)
assert exc.args == ("value message",), exc.args


# A slot the constructor fills FROM an argument must keep tracking that
# argument.  The site below is traced while `payload` is None — the arity-1
# `value` slot then reads `None`, which is indistinguishable from the default
# unless the fold looks at the argument too — and is then run with a real
# value.  `value` has to follow `args[0]` on every later iteration.
payload = None


def raise_stop_iteration_value():
    raise StopIteration(payload)


for _ in range(N):
    exc = caught(raise_stop_iteration_value, StopIteration)
assert exc.args == (None,), exc.args
assert exc.value is None, exc.value

for i in range(N):
    payload = i
    exc = caught(raise_stop_iteration_value, StopIteration)
    assert exc.value == i, (exc.value, i)
    assert exc.args == (i,), exc.args

# The same hole for `ImportError`'s lone-positional `msg`.
payload = None


def raise_import_error_msg():
    raise ImportError(payload)


for _ in range(N):
    exc = caught(raise_import_error_msg, ImportError)
assert exc.args == (None,), exc.args
assert exc.msg is None, exc.msg

for i in range(N):
    payload = str(i)
    exc = caught(raise_import_error_msg, ImportError)
    assert exc.msg == str(i), (exc.msg, i)


# `raise X(...) from Y` keeps `__cause__` and flips `__suppress_context__`.
cause = ValueError("cause")
for _ in range(N):
    try:
        raise StopIteration() from cause
    except StopIteration as exc:
        assert exc.__cause__ is cause
        assert exc.__suppress_context__ is True
        assert exc.value is None

# A raise inside an active handler chains `__context__` onto the new instance,
# so the fold's inline `__context__` store has to see the same value.
for _ in range(N):
    try:
        raise ValueError("outer")
    except ValueError as outer:
        try:
            raise AttributeError("inner")
        except AttributeError as inner:
            assert inner.__context__ is outer
            assert inner.name is None
            assert inner.args == ("inner",)

print("OK")
