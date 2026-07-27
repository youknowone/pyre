import _thread
import sys

assert _thread.TIMEOUT_MAX in [9223372036.0, 4294967.0]


ExceptHookArgs = _thread._ExceptHookArgs
assert ExceptHookArgs.__module__ == "_thread"
assert ExceptHookArgs.__name__ == "_ExceptHookArgs"
assert ExceptHookArgs.n_fields == 4
assert ExceptHookArgs.n_sequence_fields == 4
assert ExceptHookArgs.n_unnamed_fields == 0
assert ExceptHookArgs.__match_args__ == (
    "exc_type",
    "exc_value",
    "exc_traceback",
    "thread",
)
assert "Type used to pass arguments to threading.excepthook." in ExceptHookArgs.__doc__

try:
    class DerivedExceptHookArgs(ExceptHookArgs):
        pass
except TypeError:
    pass
else:
    raise AssertionError("_ExceptHookArgs unexpectedly accepted a subclass")

hook_args = ExceptHookArgs([ValueError, ValueError("thread boom"), None, None])
assert tuple(hook_args) == (ValueError, hook_args.exc_value, None, None)
assert repr(hook_args).startswith(
    "_thread._ExceptHookArgs(exc_type=<class 'ValueError'>, "
)

try:
    _thread._excepthook(())
except TypeError as exc:
    assert str(exc) == "_thread.excepthook argument type must be ExceptHookArgs"
else:
    raise AssertionError("_excepthook accepted a plain tuple")


class Sink:
    def __init__(self):
        self.parts = []
        self.flushed = False

    def write(self, text):
        self.parts.append(text)

    def flush(self):
        self.flushed = True


sink = Sink()
old_stderr = sys.stderr
try:
    sys.stderr = sink
    assert _thread._excepthook(hook_args) is None
    assert _thread._excepthook(
        ExceptHookArgs([SystemExit, SystemExit(1), None, None])
    ) is None
finally:
    sys.stderr = old_stderr

output = "".join(sink.parts)
assert output.startswith(f"Exception in thread {_thread.get_ident()}:\n")
assert output.endswith("ValueError: thread boom\n")
assert "SystemExit" not in output
assert sink.flushed

sink.parts.clear()
sink.flushed = False
try:
    raise LookupError("traceback boom")
except LookupError:
    traced_args = ExceptHookArgs([*sys.exc_info(), None])
    try:
        sys.stderr = sink
        assert _thread._excepthook(traced_args) is None
    finally:
        sys.stderr = old_stderr
        traced_args = None

output = "".join(sink.parts)
assert "Traceback (most recent call last):\n" in output
assert 'raise LookupError("traceback boom")' in output
assert output.endswith("LookupError: traceback boom\n")
assert sink.flushed


class CustomStrError(Exception):
    def __str__(self):
        return "custom thread formatting"


class BrokenStrError(Exception):
    def __str__(self):
        raise RuntimeError("str failed")


def capture_error(exc):
    try:
        raise exc
    except Exception as caught:
        return caught


def hook_output(exc, traceback=None, thread=None):
    target = Sink()
    previous = sys.stderr
    try:
        sys.stderr = target
        _thread._excepthook(
            ExceptHookArgs([type(exc), exc, traceback, thread])
        )
    finally:
        sys.stderr = previous
    assert target.flushed
    return "".join(target.parts)


# `_PyErr_Display` installs a supplied traceback only when the exception has
# none.  An exception-resident traceback wins and remains unchanged.
supplied_error = capture_error(LookupError("supplied traceback"))
supplied_tb = supplied_error.__traceback__
fresh_error = CustomStrError("raw constructor argument")
fresh_output = hook_output(fresh_error, supplied_tb)
assert fresh_error.__traceback__ is supplied_tb
assert "capture_error" in fresh_output
assert fresh_output.endswith("CustomStrError: custom thread formatting\n")

own_error = capture_error(CustomStrError("own traceback"))
own_tb = own_error.__traceback__
own_output = hook_output(own_error, supplied_tb)
assert own_error.__traceback__ is own_tb
assert own_error.__traceback__ is not supplied_tb
assert own_output.endswith("CustomStrError: custom thread formatting\n")

# Exception-only formatting calls the exception object's real `__str__`.
assert hook_output(KeyError("key")).endswith("KeyError: 'key'\n")
assert "[Errno 2] missing: 'file.txt'" in hook_output(
    OSError(2, "missing", "file.txt")
)
assert hook_output(BrokenStrError("raw")).endswith(
    "BrokenStrError: <exception str() failed>\n"
)


class BrokenThreadName:
    @property
    def name(self):
        raise NameError("thread name failed")


# `PyObject_GetOptionalAttr` suppresses AttributeError only; NameError from a
# descriptor must reach `threading`'s outer failure reporting path.
sink = Sink()
old_stderr = sys.stderr
try:
    sys.stderr = sink
    try:
        _thread._excepthook(
            ExceptHookArgs(
                [ValueError, ValueError("name"), None, BrokenThreadName()]
            )
        )
    except NameError as exc:
        assert str(exc) == "thread name failed"
    else:
        raise AssertionError("_excepthook swallowed a thread-name NameError")
finally:
    sys.stderr = old_stderr

# `_PySys_GetOptionalAttr` reads the interpreter-owned sys dictionary even
# when user code replaces the `sys.modules["sys"]` import-cache entry.
class ReplacementSys:
    pass


replacement = ReplacementSys()
wrong_sink = Sink()
replacement.stderr = wrong_sink
real_sink = Sink()
original_sys_entry = sys.modules["sys"]
old_stderr = sys.stderr
try:
    sys.modules["sys"] = replacement
    sys.stderr = real_sink
    _thread._excepthook(
        ExceptHookArgs([ValueError, ValueError("real stderr"), None, None])
    )
finally:
    sys.stderr = old_stderr
    sys.modules["sys"] = original_sys_entry
assert "ValueError: real stderr\n" in "".join(real_sink.parts)
assert wrong_sink.parts == []

import threading

assert threading.ExceptHookArgs is ExceptHookArgs
assert threading.excepthook is _thread._excepthook
