# pyre-check: gate=1
# `__pypy__.hidden_applevel` marks a code object as one the frame machinery
# hides.  Its argument is declared as a function, and a function's `getcode()`
# is not always a `PyCode`: a builtin's is a `BuiltinCode`, which has no such
# field and already reports the value the marker would write.  Writing at
# `PyCode`'s offset regardless corrupts whatever the other layout keeps there,
# so each arm below marks something whose code is not a `PyCode` and then goes
# on using it -- a wrong write shows up as a crash in the use, not in the mark.
#
# Which callables the declared argument type admits is a separate question from
# what their code's layout is, and it is a fixed one: a builtin function and a
# method descriptor are `Function`s and are marked, a slot wrapper and a bound
# builtin method are not and are declined.  Each arm says which answer it
# expects, because an arm that tolerated either would stop marking anything at
# all the moment the first question changed, and would still pass.
try:
    import __pypy__
except ImportError:  # the reference build has no such module to ask
    __pypy__ = None


def mark(obj):
    """`hidden_applevel(obj)` for an argument the declared type admits."""
    result = __pypy__.hidden_applevel(obj)
    assert result is obj, result
    return obj


def declined(obj):
    """`hidden_applevel(obj)` for an argument the declared type rejects."""
    try:
        __pypy__.hidden_applevel(obj)
    except TypeError:
        return obj
    raise AssertionError('%r was accepted' % (obj,))


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


def a_method_descriptor_keeps_working_after_the_marker():
    for descriptor in (str.upper, list.append):
        mark(descriptor)
    assert str.upper('ab') == 'AB'
    assert 'ab'.upper() == 'AB'
    holder = []
    list.append(holder, 1)
    holder.append(2)
    assert holder == [1, 2]


def a_slot_wrapper_is_declined_and_still_runs():
    for wrapper in (int.__add__, object.__init__):
        declined(wrapper)
    assert int.__add__(1, 2) == 3
    assert 1 + 2 == 3
    assert object.__init__(object()) is None


def a_bound_builtin_method_is_declined_and_still_runs():
    holder = [1]
    declined(holder.append)
    holder.append(2)
    assert holder == [1, 2]
    text = 'ab'
    declined(text.upper)
    assert text.upper() == 'AB'


if __pypy__ is not None:
    a_python_function_is_marked_and_still_runs()
    a_builtin_keeps_working_after_the_marker()
    a_method_descriptor_keeps_working_after_the_marker()
    a_slot_wrapper_is_declined_and_still_runs()
    a_bound_builtin_method_is_declined_and_still_runs()
print('OK')
