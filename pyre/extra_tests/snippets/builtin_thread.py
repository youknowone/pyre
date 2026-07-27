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


class SurrogateStrError(Exception):
    def __str__(self):
        return "bad\ud800value"


class CollectingStrError(Exception):
    def __str__(self):
        import gc

        garbage = [[index] for index in range(4096)]
        gc.collect()
        assert len(garbage) == 4096
        return "collected safely"


class BrokenNotesError(Exception):
    @property
    def __notes__(self):
        raise RuntimeError("notes failed")


class ErrorNamespace:
    class NestedError(Exception):
        pass


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
assert hook_output(SurrogateStrError()).endswith(
    "SurrogateStrError: bad\ud800value\n"
)
collecting_error = CollectingStrError()
collecting_error.add_note("note after collection")
collecting_output = hook_output(collecting_error)
assert "CollectingStrError: collected safely\n" in collecting_output
assert collecting_output.endswith("note after collection\n")

nested_output = hook_output(ErrorNamespace.NestedError("nested"))
assert nested_output.endswith("ErrorNamespace.NestedError: nested\n")
ErrorNamespace.NestedError.__module__ = "pkg.mod"
qualified_output = hook_output(ErrorNamespace.NestedError("qualified"))
assert qualified_output.endswith(
    "pkg.mod.ErrorNamespace.NestedError: qualified\n"
)

tuple_notes = ValueError("tuple notes")
tuple_notes.__notes__ = ("first detail", 42)
tuple_notes_output = hook_output(tuple_notes)
assert tuple_notes_output.endswith("first detail\n42\n")

range_notes = ValueError("range notes")
range_notes.__notes__ = range(2)
range_notes_output = hook_output(range_notes)
assert range_notes_output.endswith("0\n1\n")

surrogate_note = ValueError("surrogate note")
surrogate_note.add_note("bad\ud800note")
surrogate_note_output = hook_output(surrogate_note)
assert surrogate_note_output.endswith("bad\ud800note\n")

scalar_notes = ValueError("scalar notes")
scalar_notes.__notes__ = "detail"
scalar_notes_output = hook_output(scalar_notes)
assert scalar_notes_output.endswith("'detail'\n")

broken_notes_output = hook_output(BrokenNotesError("broken notes"))
assert broken_notes_output.endswith(
    "BrokenNotesError: broken notes\n"
    "Ignored error getting __notes__: RuntimeError('notes failed')\n"
)


def raise_name_suggestion():
    available_name = 1
    return availabl_name


try:
    raise_name_suggestion()
except NameError as suggested_name_error:
    name_suggestion_output = hook_output(suggested_name_error)
assert name_suggestion_output.endswith(
    "NameError: name 'availabl_name' is not defined. "
    "Did you mean: 'available_name'?\n"
)


class SuggestionTarget:
    available_attribute = 1


attribute_suggestion = AttributeError(
    "'SuggestionTarget' object has no attribute 'availabl_attribute'",
    name="availabl_attribute",
    obj=SuggestionTarget(),
)
assert hook_output(attribute_suggestion).endswith(
    "AttributeError: 'SuggestionTarget' object has no attribute "
    "'availabl_attribute'. Did you mean: 'available_attribute'?\n"
)


class PrivateSuggestionTarget:
    _public = 1

    def __dir__(self):
        import gc

        garbage = [[index] for index in range(4096)]
        gc.collect()
        assert len(garbage) == 4096
        return ["_public"]

    def fail(self):
        return self.public


try:
    PrivateSuggestionTarget().fail()
except AttributeError as private_suggestion:
    private_suggestion_output = hook_output(private_suggestion)
assert private_suggestion_output.endswith(
    "AttributeError: 'PrivateSuggestionTarget' object has no attribute "
    "'public'. Did you mean: '_public'?\n"
)

import_suggestion = ImportError(
    "cannot import name 'sqr' from 'math'",
    name="math",
    name_from="sqr",
)
import_suggestion_output = hook_output(import_suggestion)
assert import_suggestion_output.endswith(
    "ImportError: cannot import name 'sqr' from 'math'. Did you mean: 'sqrt'?\n"
), repr(import_suggestion_output)

# CPython's invalid-value printer intentionally retains its historical
# `NoneType: None` special case while other non-exceptions use the diagnostic.
assert hook_output(None).endswith("NoneType: None\n")
assert hook_output(42).endswith(
    "TypeError: print_exception(): Exception expected for value, int found\n"
)

located_syntax_error = SyntaxError(
    "bad syntax", ("thread_syntax.py", 3, 4, "abc\n", 3, 5)
)
syntax_output = hook_output(located_syntax_error)
assert 'File "thread_syntax.py", line 3\n' in syntax_output
assert "SyntaxError: bad syntax\n" in syntax_output
assert "(thread_syntax.py, line 3)" not in syntax_output

group_output = hook_output(
    ExceptionGroup(
        "outer",
        [
            CollectingStrError(),
            ExceptionGroup("inner", [TypeError("nested leaf")]),
        ],
    )
)
assert "ExceptionGroup: outer (2 sub-exceptions)\n" in group_output
assert "CollectingStrError: collected safely\n" in group_output
assert "ExceptionGroup: inner (1 sub-exception)\n" in group_output
assert "TypeError: nested leaf\n" in group_output
assert "+---------------- 1 ----------------\n" in group_output
assert "+------------------------------------\n" in group_output

later_group_child = TypeError("later child")
first_group_child = ValueError("first child")
first_group_child.__cause__ = later_group_child
sibling_chain_group_output = hook_output(
    ExceptionGroup("sibling chain", [first_group_child, later_group_child])
)
assert sibling_chain_group_output.count("TypeError: later child\n") == 1
assert "direct cause" not in sibling_chain_group_output

self_cause = ValueError("self cycle")
self_cause.__cause__ = self_cause
self_cause_output = hook_output(self_cause)
assert self_cause_output.count("ValueError: self cycle\n") == 1
assert "direct cause" not in self_cause_output

cycle_first = ValueError("cycle first")
cycle_second = TypeError("cycle second")
cycle_first.__context__ = cycle_second
cycle_second.__context__ = cycle_first
context_cycle_output = hook_output(cycle_first)
assert context_cycle_output.count("ValueError: cycle first\n") == 1
assert context_cycle_output.count("TypeError: cycle second\n") == 1
assert context_cycle_output.count("During handling") == 1

deep_chain = ValueError("chain 0")
for chain_index in range(1, 1200):
    next_error = ValueError(f"chain {chain_index}")
    next_error.__context__ = deep_chain
    deep_chain = next_error
deep_chain_output = hook_output(deep_chain)
assert deep_chain_output.count("During handling") == 1199
assert "ValueError: chain 0\n" in deep_chain_output
assert "ValueError: chain 1199\n" in deep_chain_output


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


class FakeThread:
    def __init__(self, name, stderr):
        self.name = name
        self._stderr = stderr


named_thread = FakeThread("worker-1", Sink())
assert hook_output(ValueError("named"), thread=named_thread).startswith(
    "Exception in thread worker-1:\n"
)

# Exercise the saved `_stderr` fallback.  CPython 3.14's stdlib traceback
# fast path writes the exception body through `sys.__stderr__`, while its
# C fallback (and pyre) writes it to the explicit file; the named banner and
# flush prove that `_thread` selected the thread-owned stream in both cases.
fallback_sink = Sink()
backup_sink = Sink()
fallback_thread = FakeThread("worker-fallback", fallback_sink)
old_stderr = sys.stderr
old_dunder_stderr = sys.__stderr__
try:
    sys.stderr = None
    sys.__stderr__ = backup_sink
    _thread._excepthook(
        ExceptHookArgs(
            [ValueError, ValueError("fallback"), None, fallback_thread]
        )
    )
finally:
    sys.stderr = old_stderr
    sys.__stderr__ = old_dunder_stderr
assert "".join(fallback_sink.parts).startswith(
    "Exception in thread worker-fallback:\n"
)
assert fallback_sink.flushed

import threading

assert threading.ExceptHookArgs is ExceptHookArgs
assert threading.excepthook is _thread._excepthook
