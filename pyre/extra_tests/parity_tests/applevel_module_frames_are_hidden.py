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
# Every name below is one PyPy hides too, so this file is checked against PyPy
# as well.  The two subjects where PyPy answers differently are split into
# `contextvars_and_hash_bodies_are_not_frames.py`, which says so in its header.
import sys
import atexit
import collections
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

    # Bundled app-level source resolves its own sys module without consulting
    # a program's blocked import entry.
    class Slotted:
        __slots__ = ('value',)

    slotted = Slotted()
    slotted.value = 1
    saved = sys.modules['sys']
    sys.modules['sys'] = None
    try:
        reduction = slotted.__reduce_ex__(2)
    finally:
        sys.modules['sys'] = saved
    assert reduction[1] == (Slotted,)
    assert reduction[2] == (None, {'value': 1})


def the_defaultdict_factory_is_not_a_frame():
    d = collections.defaultdict(int)
    # A miss fills the key it was asked for, and `traced` runs its callable once
    # outside the trace, so a fixed key would leave the traced call an ordinary
    # hit that never reaches the factory -- and the arm would pass however
    # visible that frame became.  Hand each call a key nothing has filled.
    keys = iter(('warm', 'traced'))
    seen = traced(lambda: d[next(keys)])
    assert seen == ['<lambda>'], seen
    assert d['traced'] == 0, d


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
a_failure_inside_aiter_records_no_frame()
the_caller_a_type_parameter_records_is_this_frame()
print('OK')
