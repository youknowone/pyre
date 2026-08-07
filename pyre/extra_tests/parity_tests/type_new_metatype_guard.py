"""`type.__new__` argument handling, including a non-type metatype.

`descr__new__` (typeobject.py:886-911) decides the arity first, then runs
`_precheck_for_new` (typeobject.py:1001-1003), so a call carrying one
name argument reports the metatype as not being a type, while a call
carrying none reports the arity under the `%N` operand spelling —
`W_Root.getname` (baseobjspace.py:90-94), which answers `?` when the
object has no `__name__`.

The precheck is unconditional, so the three-argument form is covered too:
the metatype is a parameter beside `__args__.arguments_w` rather than one of
them, which fixes its position at every arity.

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

# The three-argument form runs the same precheck.  `descr__new__` takes the
# metatype as a parameter beside `__args__.arguments_w`, so it is `pos[0]`
# whatever the arity; keying the (name, bases, dict) triple off argument
# *types* instead lets these three build a class.
expect_type_error(
    "int_metatype_three_args",
    lambda: type.__new__(42, "A", (), {}),
    "X is not a type object (int)",
    "type.__new__(X): X is not a type object (int)",
)
expect_type_error(
    "none_metatype_three_args",
    lambda: type.__new__(None, "A", (), {}),
    "X is not a type object (NoneType)",
    "type.__new__(X): X is not a type object (NoneType)",
)
expect_type_error(
    "str_metatype_three_args",
    lambda: type.__new__("s", "A", (), {}),
    "X is not a type object (str)",
    "type.__new__(X): X is not a type object (str)",
)


class SuperMeta(type):
    def __new__(mcls, name, bases, ns):
        return super().__new__(mcls, name, bases, ns)


class ViaSuper(metaclass=SuperMeta):
    marker = 17


# `super().__new__` resolves to the very same carrier as `type.__new__` and
# prepends no receiver, so its metatype is `pos[0]` too.
expect_value("super_new_is_type_new", lambda: SuperMeta.__mro__[1].__new__, type.__new__)
expect_value("super_new_metaclass", lambda: type(ViaSuper), SuperMeta)
expect_value("super_new_body", lambda: ViaSuper.marker, 17)

# The `_ast` heap types are built by an in-tree caller that hands
# `type.__new__` its arguments directly, so they are the one construction path
# where the metatype is not supplied by an attribute lookup.
import ast

expect_value("ast_root_metatype", lambda: type(ast.AST), type)
expect_value("ast_node_metatype", lambda: type(ast.Module), type)
expect_value("ast_node_name", lambda: ast.Module.__name__, "Module")
expect_value("ast_node_base", lambda: issubclass(ast.Module, ast.AST), True)
expect_value("ast_parse_result", lambda: isinstance(ast.parse("x = 1"), ast.Module), True)

print("OK")
