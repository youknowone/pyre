"""`type.__new__` argument handling, including a non-type metatype.

`descr__new__` (typeobject.py:886-911) decides the arity first, then runs
`_precheck_for_new` (typeobject.py:1001-1003), so a call carrying one
name argument reports the metatype as not being a type, while a call
carrying none reports the arity under the `%N` operand spelling —
`W_Root.getname` (baseobjspace.py:90-94), which answers `?` when the
object has no `__name__`.

cpython words all of these differently, so only the outcome kind is
compared for the rows where it does; the messages are asserted per
runtime.
"""

import sys

IS_CPYTHON = sys.implementation.name == "cpython"


def expect_type_error(label, fn, message, cpython_message):
    try:
        fn()
    except TypeError as exc:
        expected = cpython_message if IS_CPYTHON else message
        assert str(exc) == expected, (label, str(exc))
    except BaseException as exc:
        raise AssertionError((label, type(exc).__name__, str(exc)))
    else:
        raise AssertionError((label, "no TypeError"))


def expect_value(label, fn, value, cpython_message=None):
    """`value` is what pypy and pyre answer; cpython may refuse instead."""
    try:
        result = fn()
    except TypeError as exc:
        assert IS_CPYTHON and cpython_message is not None, (label, str(exc))
        assert str(exc) == cpython_message, (label, str(exc))
    else:
        assert not (IS_CPYTHON and cpython_message is not None), (label, result)
        assert result == value, (label, result)


# A non-type metatype is named, not read as a type.  Reading it as one
# segfaulted on an int and reported the str's own bytes as a type name.
expect_type_error(
    "int_metatype",
    lambda: type.__new__(42, 1),
    "X is not a type object (int)",
    "type.__new__(X): X is not a type object (int)",
)
expect_type_error(
    "str_metatype",
    lambda: type.__new__("s", 1),
    "X is not a type object (str)",
    "type.__new__(X): X is not a type object (str)",
)
expect_type_error(
    "none_metatype",
    lambda: type.__new__(None, 1),
    "X is not a type object (NoneType)",
    "type.__new__(X): X is not a type object (NoneType)",
)

# No name argument: the arity is reported before the metatype is checked,
# so an int metatype reaches `%N` and answers `?`.
expect_type_error(
    "int_metatype_no_name",
    lambda: type.__new__(42),
    "?.__new__() takes exactly 3 arguments (1 given)",
    "type.__new__(X): X is not a type object (int)",
)
expect_type_error(
    "type_metatype_alone",
    lambda: type.__new__(type),
    "type.__new__() takes 1 or 3 arguments",
    "type.__new__() takes exactly 3 arguments (0 given)",
)

# A real metatype that is not `type` itself keeps naming the three-argument
# form; `type` itself keeps answering the one-argument form.
expect_type_error(
    "int_class_metatype",
    lambda: type.__new__(int, 1),
    "int.__new__() takes exactly 3 arguments (1 given)",
    "type.__new__(int): int is not a subtype of type",
)
expect_value(
    "type_of_one",
    lambda: type.__new__(type, 1),
    int,
    "type.__new__() takes exactly 3 arguments (1 given)",
)

# The ordinary forms are untouched.
expect_value("type_int", lambda: type(42), int)
expect_value("type_type", lambda: type(int), type)
expect_value("type_str", lambda: type("x"), str)
expect_value("type_three", lambda: type("C", (), {}).__name__, "C")


class Plain:
    pass


class Meta(type):
    pass


class WithMeta(metaclass=Meta):
    pass


expect_value("class_plain", lambda: Plain.__name__, "Plain")
expect_value("class_metaclass", lambda: type(WithMeta), Meta)

print("OK")
