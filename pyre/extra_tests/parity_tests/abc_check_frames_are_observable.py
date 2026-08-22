# CPython-suite gap: `test_abc` never installs a tracer around an ABC check,
# and no suite test walks the traceback of a failure raised behind one, so
# nothing there notices if the forwarding frames stop existing.
# `ABCMeta.__instancecheck__` and `__subclasscheck__` are ordinary Python
# functions in `abc.py`, so each ABC check pushes a frame a program can see.
# `_abc_instancecheck` is native here, and answering the check natively one
# level higher -- straight from `isinstance`, skipping the method that forwards
# to it -- is a standing performance idea: the frame is a third of what an ABC
# check costs on this runtime.
#
# parity-tests reason: that frame is not an implementation detail.  A tracer
# installed with `sys.settrace` receives a `call` event for it, and a failure
# raised behind the check reports it in the traceback, so removing it changes
# what a debugger shows and what an error says.  Both are pinned here because
# neither is visible from a timing or a wrong answer -- a shortcut that skipped
# the frame would pass every other test in this suite.
#
# CPython, PyPy and pyre agree on both observations today, so this is a parity
# fact rather than a house rule.
import sys
import abc


def instancecheck_is_a_traced_call():
    class Shape(metaclass=abc.ABCMeta):
        pass

    class Circle(Shape):
        pass

    circle = Circle()
    # Warm the positive cache, so what the tracer sees is the *fast* path --
    # the one a shortcut would take over.
    assert isinstance(circle, Shape) is True

    seen = []

    def record(frame, event, arg):
        if event == 'call':
            seen.append(frame.f_code.co_name)
        return None

    sys.settrace(record)
    try:
        isinstance(circle, Shape)
    finally:
        sys.settrace(None)

    assert '__instancecheck__' in seen, seen


def a_failure_behind_the_check_names_the_frames():
    class Refusing(metaclass=abc.ABCMeta):
        @classmethod
        def __subclasshook__(cls, subclass):
            raise ValueError('refused')

    class Probe:
        pass

    try:
        isinstance(Probe(), Refusing)
    except ValueError:
        traceback = sys.exc_info()[2]
    else:
        raise AssertionError('the hook did not raise')

    names = []
    while traceback is not None:
        names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    # The check reaches the hook through both forwarding methods, and each one
    # is a frame the report walks.
    assert names.count('__instancecheck__') == 1, names
    assert names.count('__subclasscheck__') == 1, names
    assert names[-1] == '__subclasshook__', names


def the_cache_membership_body_is_not_a_frame():
    class Shape(metaclass=abc.ABCMeta):
        pass

    class Circle(Shape):
        pass

    circle = Circle()
    # A populated positive cache, so the membership test really runs.
    assert isinstance(circle, Shape) is True

    seen = []

    def record(frame, event, arg):
        if event == 'call':
            seen.append(frame.f_code.co_name)
        return None

    sys.settrace(record)
    try:
        isinstance(circle, Shape)
    finally:
        sys.settrace(None)

    # The opposite pin to the two above, and the reason they are not a blanket
    # rule about everything an ABC check touches.  The cache is a weak-set
    # whose membership test is written in Python on one of these runtimes, yet
    # no runtime reports a frame for it: where it is a built-in it has none,
    # and where it is the module's own helper source the frame is hidden --
    # `ApplevelClass.hidden_applevel` marks that code, `frame.hide()` drops it
    # from the trace hook and `record_application_traceback` refuses to record
    # it.  So answering membership without a frame is what every runtime
    # already does, and only the `abc.py` frames above are observable.
    assert seen.count('__contains__') == 0, seen

    class Raising(type):
        def __hash__(cls):
            raise ValueError('refused')

    class Opaque(metaclass=Raising):
        pass

    try:
        isinstance(Opaque(), Shape)
    except ValueError:
        traceback = sys.exc_info()[2]
    else:
        raise AssertionError('the hash did not raise')

    names = []
    while traceback is not None:
        names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    assert names.count('__contains__') == 0, names
    assert names[-1] == '__hash__', names


instancecheck_is_a_traced_call()
a_failure_behind_the_check_names_the_frames()
the_cache_membership_body_is_not_a_frame()
print('OK')
