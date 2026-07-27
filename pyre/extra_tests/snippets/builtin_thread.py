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

import threading

assert threading.ExceptHookArgs is ExceptHookArgs
assert threading.excepthook is _thread._excepthook
