# pyre-check: gate=1
# `__pypy__.hidden_applevel` marks a code object as one the frame machinery
# hides.  Its argument is declared as a function, and a function's `getcode()`
# is not always a `PyCode`: a builtin's is a `BuiltinCode`, which has no such
# field and already reports the value the marker would write.  Writing at
# `PyCode`'s offset regardless corrupts whatever the other layout keeps there,
# so each arm below marks something whose code is not a `PyCode` and then goes
# on using it -- a wrong write shows up as a crash in the use, not in the mark.
try:
    import __pypy__
except ImportError:  # the reference build has no such module to ask
    __pypy__ = None


def mark(obj):
    """`hidden_applevel(obj)`, tolerating the argument check declining it.

    Whether a given callable is a `Function` at all is a separate question from
    what its code's layout is; this file is about the second, so an arm that
    the first one rejects still counts as covered.
    """
    try:
        result = __pypy__.hidden_applevel(obj)
    except TypeError:
        return obj
    assert result is obj, result
    return obj


def a_python_function_is_marked_and_still_runs():
    def body():
        return 7

    assert mark(body) is body
    assert body() == 7
    assert body.__code__.co_name == 'body'


def a_builtin_keeps_working_after_the_marker():
    for builtin in (len, repr, isinstance, getattr):
        mark(builtin)
    assert len('abc') == 3
    assert repr(1) == '1'
    assert isinstance(1, int) is True
    assert getattr(1, 'real') == 1


def a_type_slot_wrapper_keeps_working_after_the_marker():
    for callable_ in (str.upper, list.append, int.__add__, object.__init__):
        mark(callable_)
    assert 'ab'.upper() == 'AB'
    holder = []
    holder.append(1)
    assert holder == [1]
    assert int.__add__(1, 2) == 3


def a_bound_builtin_method_keeps_working_after_the_marker():
    holder = [1]
    mark(holder.append)
    holder.append(2)
    assert holder == [1, 2]
    text = 'ab'
    mark(text.upper)
    assert text.upper() == 'AB'


if __pypy__ is not None:
    a_python_function_is_marked_and_still_runs()
    a_builtin_keeps_working_after_the_marker()
    a_type_slot_wrapper_keeps_working_after_the_marker()
    a_bound_builtin_method_keeps_working_after_the_marker()
print('OK')
