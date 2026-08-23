# CPython-suite gap: nothing in the suite installs a tracer around a builtin
# whose implementation happens to be written in Python on some runtime, or
# walks the traceback of a failure raised inside one, so nothing there notices
# when such a helper starts reporting a frame.  On CPython every name below
# lives in an extension module and has no frame to report, which is exactly why
# the suite cannot express the question.
#
# parity-tests reason: a frame is not an implementation detail once a program
# can see it.  `sys.settrace` receives a `call` event for it, a traceback
# records it, and `sys._getframe` counts along a walk that includes it -- so a
# helper that is native on CPython and app-level here must still answer all
# three the same way.  Every arm is a silent difference in what a debugger
# shows rather than a wrong answer or a crash.
#
# PyPy has the same problem and the same mechanism: a mixed module's app-level
# half loads through `gateway.ApplevelClass`, whose `hidden_applevel` marks the
# code, and `PyFrame.hide` then keeps those frames out of the trace hook
# (`executioncontext.py _trace`), out of `record_application_traceback` and out
# of `getnextframe_nohidden`.
#
# Where PyPy does NOT apply it, the visible frames are an artefact of PyPy
# implementing in Python (in `lib_pypy`) what CPython implements in C, not a
# decision that those frames belong to the program: `_contextvars.Context.run`
# only became hidden in PyPy 7.3.14, reactively, after it broke Django.  So the
# arms below follow CPython, and `contextvars` and `blake2b` in particular
# report FEWER frames here than a PyPy 3.11 build does.
import sys
import atexit
import collections
import contextvars
import hashlib
import operator
import typing


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


def the_object_reduce_helpers_are_not_frames():
    class Simple:
        pass

    obj = Simple()
    # The protocol-2 reduction walks the instance dict and the slot names, and
    # where that walk is written in Python it is still not the program's.
    assert traced(lambda: obj.__reduce_ex__(2)) == ['<lambda>'], traced(
        lambda: obj.__reduce_ex__(2)
    )


def the_defaultdict_factory_is_not_a_frame():
    d = collections.defaultdict(int)
    assert traced(lambda: d['missing']) == ['<lambda>'], traced(lambda: d['missing'])


def the_operator_getters_are_not_frames():
    get = operator.attrgetter('real')
    assert traced(lambda: get(1)) == ['<lambda>'], traced(lambda: get(1))
    assert traced(lambda: operator.itemgetter(0)([7])) == ['<lambda>']


def the_atexit_registry_is_not_a_frame():
    def callback():
        pass

    def register_and_drop():
        atexit.register(callback)
        atexit.unregister(callback)

    assert traced(register_and_drop) == ['register_and_drop'], traced(register_and_drop)


def the_context_machinery_is_not_frames():
    var = contextvars.ContextVar('probe', default=0)
    ctx = contextvars.copy_context()

    def run_in_context():
        return ctx.run(lambda: var.get())

    # `run`, the variable lookup and the persistent map that stores the
    # bindings are one extension module on CPython, so the only frame this
    # reports is the callable the context was asked to run.
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


def a_failure_inside_aiter_records_no_frame():
    class NotAnAsyncIterator:
        def __aiter__(self):
            return self

    try:
        aiter(NotAnAsyncIterator())
    except TypeError:
        traceback = sys.exc_info()[2]
    else:
        raise AssertionError('aiter accepted a non-async-iterator')

    names = []
    while traceback is not None:
        names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    assert names == ['a_failure_inside_aiter_records_no_frame'], names


def the_caller_a_type_parameter_records_is_this_frame():
    # The arm that fails in the other direction.  A type-parameter object takes
    # its `__module__` from the frame that is running when it is built, and
    # every frame between this one and that read is one of the hidden ones --
    # so the answer has to be this module, not whoever called it.
    assert typing.TypeVar('T').__module__ == '__main__', typing.TypeVar('T').__module__
    assert typing.ParamSpec('P').__module__ == '__main__'
    assert typing.TypeVarTuple('Ts').__module__ == '__main__'

    def built_one_frame_deeper():
        return typing.TypeVar('U').__module__

    assert built_one_frame_deeper() == '__main__', built_one_frame_deeper()


the_object_reduce_helpers_are_not_frames()
the_defaultdict_factory_is_not_a_frame()
the_operator_getters_are_not_frames()
the_atexit_registry_is_not_a_frame()
the_context_machinery_is_not_frames()
the_hash_wrappers_are_not_frames()
a_failure_inside_aiter_records_no_frame()
the_caller_a_type_parameter_records_is_this_frame()
print('OK')
