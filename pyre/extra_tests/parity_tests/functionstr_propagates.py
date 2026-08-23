# CPython-suite gap: `test_extcall` and `test_call` check the `f() argument
# after * must be an iterable` message, but only ever for callables whose
# `__qualname__` and `__module__` are ordinary strings.  A runtime that
# swallowed every fault while building that prefix would pass both modules.
#
# parity-tests reason: the prefix comes from `_PyObject_FunctionStr`, which is
# reached only while another error is already being reported.  That makes it
# tempting to treat it as best-effort and fall back to `str(x)` on any
# problem -- and PyPy does exactly that, with two
# `try/except OperationError: pass` blocks.  3.14 propagates instead: a
# `__qualname__` descriptor that raises replaces the whole message with its
# own error.  Nothing else in the suite forces that choice.
#
# The prefix rule is a rich comparison, not a type check: `__module__` is
# rendered with `%S` whenever it is neither `None` nor equal to `'builtins'`,
# so a non-string one is formatted rather than dropped, and `''` yields a bare
# leading dot.
#
# PyPy 7.3.20 fails every case here.  Its `except OperationError: pass` sends
# the class cells to the `getname() + ' object'` answer and leaves the plain
# functions with a bare `f()`, so every raising descriptor is swallowed and
# every module that is not a plain non-empty `str` is dropped.
import re
import sys


def message(callable_object):
    """The report from calling `callable_object` with a non-iterable `*arg`."""
    try:
        callable_object(*1)
    except BaseException as exc:
        # The `str(x)` fallback embeds an address; nothing here should reach it.
        return type(exc).__name__ + ': ' + re.sub(r'0x[0-9a-fA-F]+', '0xADDR', str(exc))
    return 'no error'


def named(qualname):
    """A callable answering `qualname` for `__qualname__` and nothing else."""

    class C:
        def __call__(self, *args):
            pass

        def __getattr__(self, name):
            if name == '__qualname__':
                return qualname
            raise AttributeError(name)

    return C


def a_raising_qualname_replaces_the_message():
    class C:
        def __call__(self, *args):
            pass

        def __getattr__(self, name):
            raise TypeError(name + ' boom')

    print('qualname raises:', message(C()))


def a_non_string_qualname_is_formatted_not_dropped():
    print('qualname is 42:', message(named(42)()))


def the_module_test_is_a_comparison_not_a_type_check():
    for module in (42, None, 'builtins', '', 'mymod'):
        cls = named('Q')
        cls.__module__ = module
        print('module %-10r ->' % (module,), message(cls()))


def a_raising_str_on_either_name_replaces_the_message():
    class StrRaises:
        def __str__(self):
            raise TypeError('str boom')

    class ModuleStrRaises:
        def __str__(self):
            raise TypeError('module str boom')

    cls = named('Q')
    cls.__module__ = StrRaises()
    print('module __str__ raises:', message(cls()))
    print('qualname __str__ raises:', message(named(StrRaises())()))
    # `%S.%S()` converts the module first, so its error is the one that wins.
    cls = named(StrRaises())
    cls.__module__ = ModuleStrRaises()
    print('both __str__ raise:', message(cls()))


def the_absent_qualname_fallback_is_str_and_str_can_fail():
    class StrRaises:
        def __call__(self, *args):
            pass

        def __str__(self):
            raise TypeError('str boom')

    class StrNonString:
        def __call__(self, *args):
            pass

        def __str__(self):
            return 42

    print('no qualname, __str__ raises:', message(StrRaises()))
    print('no qualname, __str__ is int:', message(StrNonString()))


def the_module_comparison_is_ne_and_reaches_a_str_subclass():
    """`PyObject_RichCompareBool(module, 'builtins', Py_NE)` — `__ne__`, and no
    exact-type shortcut around it.

    A runtime that answers the module test by reading the string's bytes
    whenever it is *a* str takes that shortcut for a str SUBCLASS too, and
    then a subclass overriding `__ne__` never runs.  Both cells below stop the
    message with the subclass's own error, including the one whose text is
    exactly `'builtins'`.

    The `__eq__`-vs-`__ne__` half is only visible when the two disagree, which
    is why each class here raises a differently-worded error from each.
    """

    class NeRaises:
        def __ne__(self, other):
            raise TypeError('__ne__ boom')

        def __eq__(self, other):
            raise TypeError('__eq__ boom')

        __hash__ = None

        def __str__(self):
            return 'nemod'

    class SubNeRaises(str):
        def __ne__(self, other):
            raise TypeError('sub __ne__ boom')

        def __eq__(self, other):
            raise TypeError('sub __eq__ boom')

        __hash__ = str.__hash__

    def f(*args):
        pass

    f.__module__ = NeRaises()
    print('module __ne__ raises:', message(f))
    for text in ('mymod', 'builtins'):
        f.__module__ = SubNeRaises(text)
        print('str-subclass module %-10r ->' % (text,), message(f))


def a_function_takes_the_same_module_rule():
    def f(*args):
        pass

    for module in ('', 42, None, 'builtins'):
        f.__module__ = module
        print('function module %-10r ->' % (module,), message(f))


a_raising_qualname_replaces_the_message()
a_non_string_qualname_is_formatted_not_dropped()
the_module_test_is_a_comparison_not_a_type_check()
a_raising_str_on_either_name_replaces_the_message()
the_absent_qualname_fallback_is_str_and_str_can_fail()
a_function_takes_the_same_module_rule()
the_module_comparison_is_ne_and_reaches_a_str_subclass()
print('OK')
