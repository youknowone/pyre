# pyre-check: pypy-diverges: pins the frameless contextvars and blake2b of a C implementation; pypy3 writes both in Python and shows their frames
# CPython-suite gap: the suite never installs a tracer around a context switch
# or a hash update, because on CPython there is nothing there to trace -- both
# are extension modules in 3.11 and in 3.14 alike.
#
# parity-tests reason: pyre writes these two module bodies in Python, so
# without a marker on that code their frames reach `sys.settrace`, a recorded
# traceback and the walk `sys._getframe` counts along.  None of that shows up
# as a wrong answer, so only a frame-shaped test can hold the line.
#
# Split out of `applevel_module_frames_are_hidden.py` because these two arms
# are the ones PyPy answers differently, and the rest of that file is checked
# against PyPy as well.  PyPy also writes them in Python -- in `lib_pypy` --
# and leaves the frames visible, which is a consequence of the implementation
# rather than a decision about them: `Context.run` alone became hidden in PyPy
# 7.3.14, reactively, after it broke Django (release note: "Discovered in
# django PR 17500").  So the reference here is CPython.
import sys
import contextvars
import hashlib


def traced(call):
    seen = []

    def record(frame, event, arg):
        if event == 'call':
            seen.append(frame.f_code.co_name)
        return None

    call()  # warm every lazy import and cache outside the trace
    sys.settrace(record)
    try:
        call()
    finally:
        sys.settrace(None)
    return seen


def the_context_machinery_is_not_frames():
    var = contextvars.ContextVar('probe', default=0)
    ctx = contextvars.copy_context()

    def run_in_context():
        return ctx.run(lambda: var.get())

    # `run`, the variable lookup and the persistent map holding the bindings
    # are one extension module on CPython, so the only frame this reports is
    # the callable the context was asked to run.
    assert traced(run_in_context) == ['run_in_context', '<lambda>'], traced(run_in_context)

    def set_and_reset():
        token = var.set(1)
        var.reset(token)

    assert traced(set_and_reset) == ['set_and_reset'], traced(set_and_reset)


def the_hash_wrappers_are_not_frames():
    assert traced(lambda: hashlib.blake2b(b'abc').hexdigest()) == ['<lambda>'], traced(
        lambda: hashlib.blake2b(b'abc').hexdigest()
    )
    assert traced(lambda: hashlib.sha256(b'abc').hexdigest()) == ['<lambda>']


the_context_machinery_is_not_frames()
the_hash_wrappers_are_not_frames()
print('OK')
