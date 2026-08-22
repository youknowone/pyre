# CPython-suite gap: nothing in the suite installs a tracer around a builtin
# whose implementation happens to be written in Python on some runtime, or
# walks the traceback of a failure raised inside one, so nothing there notices
# when such a helper starts reporting a frame.  On CPython every name below is
# native and has no frame to report, which is exactly why the suite cannot
# express the question.
#
# parity-tests reason: a frame is not an implementation detail once a program
# can see it.  `sys.settrace` receives a `call` event for it and a traceback
# records it, so a helper that is native on one runtime and app-level on
# another must still answer both the same way -- and every arm here is a silent
# difference in what a debugger shows rather than a wrong answer or a crash.
#
# PyPy writes several of these in Python and keeps them invisible by
# construction: a mixed module's app-level half loads through
# `gateway.ApplevelClass`, whose `hidden_applevel` marks the code, and
# `PyFrame.hide` then keeps those frames out of the trace hook
# (`executioncontext.py _trace`), out of `record_application_traceback`, and
# out of the `getnextframe_nohidden` walk.  The last arm is the other half of
# that rule: a module PyPy ships as an ordinary import is the program's own,
# its frames stay visible, and one of them counts them with `sys._getframe`.
import sys
import atexit
import collections
import contextvars
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


def the_context_run_body_is_not_a_frame():
    ctx = contextvars.copy_context()
    names = traced(lambda: ctx.run(lambda: None))
    # `run` is the frame between a callable and whoever asked the context to
    # run it, and it is marked one at a time rather than by its module: the
    # rest of the context object is the program's own.
    assert names.count('run') == 0, names
    assert names.count('<lambda>') == 2, names


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


def a_module_the_program_imports_keeps_its_frames():
    # The opposite pin.  A type-parameter object takes its `__module__` from
    # the caller by counting frames back to it, so the helper's own frames and
    # the constructor's have to be there to be counted.  Hiding the module that
    # defines them would move the count onto this file's caller instead.
    assert typing.TypeVar('T').__module__ == '__main__', typing.TypeVar('T').__module__
    assert typing.ParamSpec('P').__module__ == '__main__'
    assert typing.TypeVarTuple('Ts').__module__ == '__main__'


the_object_reduce_helpers_are_not_frames()
the_defaultdict_factory_is_not_a_frame()
the_operator_getters_are_not_frames()
the_atexit_registry_is_not_a_frame()
the_context_run_body_is_not_a_frame()
a_failure_inside_aiter_records_no_frame()
a_module_the_program_imports_keeps_its_frames()
print('OK')
