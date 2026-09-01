# CPython-suite gap: exception tests do not hot-loop flattened slots after bare-class raises.
# parity-tests reason: pins slot defaults and live payloads without distorting synth jitstats.

"""Bare-class raise initializes every flattened exception slot."""

def stop_iteration():
    raise StopIteration


def name_error():
    raise NameError


def attribute_error():
    raise AttributeError


def import_error():
    raise ImportError

def caught(fn, exc_type):
    try:
        fn()
    except exc_type as exc:
        return exc
    raise AssertionError("expected exception")


payload = None


def stop_value():
    raise StopIteration(payload)


def import_value():
    raise ImportError(payload)


def module_not_found():
    raise ModuleNotFoundError


def value_error():
    raise ValueError


def attribute_value():
    raise AttributeError("attr message")


def name_value():
    raise NameError("name message")


def import_pair():
    raise ImportError("import message", "second")


for _ in range(3000):
    stop = caught(stop_iteration, StopIteration)
    assert stop.value is None and stop.args == ()
    assert stop.__cause__ is None and stop.__context__ is None
    assert stop.__suppress_context__ is False and stop.__traceback__ is not None
    name = caught(name_error, NameError)
    assert name.name is None and name.args == ()
    attr = caught(attribute_error, AttributeError)
    assert (attr.name, attr.obj, attr.args) == (None, None, ())
    imp = caught(import_error, ImportError)
    assert (imp.name, imp.path, imp.msg, imp.args) == (None, None, None, ())
    missing = caught(module_not_found, ModuleNotFoundError)
    assert (missing.name, missing.path, missing.msg, missing.args) == (None, None, None, ())
    assert caught(value_error, ValueError).args == ()
    attr = caught(attribute_value, AttributeError)
    assert (attr.args, attr.name, attr.obj) == (("attr message",), None, None)
    name = caught(name_value, NameError)
    assert (name.args, name.name) == (("name message",), None)
    imp = caught(import_pair, ImportError)
    assert (imp.args, imp.name, imp.path, imp.msg) == (
        ("import message", "second"), None, None, None
    )
    assert caught(stop_value, StopIteration).value is None
    assert caught(import_value, ImportError).msg is None

for payload in range(3000):
    stop = caught(stop_value, StopIteration)
    assert stop.value == payload and stop.args == (payload,)
    text = str(payload)
    payload = text
    imp = caught(import_value, ImportError)
    assert imp.msg == text and imp.args == (text,)

cause = ValueError("cause")
for _ in range(3000):
    try:
        raise StopIteration from cause
    except StopIteration as stop:
        assert stop.__cause__ is cause and stop.__suppress_context__ is True
        assert stop.value is None
    try:
        raise ValueError("outer")
    except ValueError as outer:
        try:
            raise StopIteration
        except StopIteration as inner:
            assert inner.__context__ is outer and inner.value is None

print("OK")
