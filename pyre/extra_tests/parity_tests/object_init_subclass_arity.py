"""`object.__init_subclass__` refuses what its declared signature refuses.

`descr___init_subclass__(space, w_cls)` (objectobject.py:139) declares one
parameter and no `__args__`, so everything past the bound class is refused by
the argument parser before the body runs — and the parser's `ArgErr` shapes
(argument.py:529-627) are what word the refusal.  Two consequences the ad-hoc
"takes no keyword arguments" spelling does not have: an extra *positional*
argument is an error too, and the message names the defining type, so it reads
`object.__init_subclass__()` even when the call arrives through a subclass or
through a class statement's keywords.

cpython refuses the same calls under `tp_new_wrapper`-style wording and names
the *receiving* class, so every message is asserted per runtime.
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


# The bound class alone is the whole accepted signature.
assert object.__init_subclass__() is None
assert int.__init_subclass__() is None


# An extra positional argument is refused.  pyre used to return None here,
# which is the row that makes this more than a wording difference.
expect_type_error(
    "one_extra_positional",
    lambda: object.__init_subclass__(1),
    "object.__init_subclass__() takes 1 positional argument but 2 were given. "
    "Did you forget 'self' in the function definition?",
    "object.__init_subclass__() takes no arguments (1 given)",
)
# The "did you forget self?" hint is conditional on `given == num_argnames + 1`
# (argument.py:592-594), so a second extra argument drops it.
expect_type_error(
    "two_extra_positional",
    lambda: object.__init_subclass__(1, 2),
    "object.__init_subclass__() takes 1 positional argument but 3 were given",
    "object.__init_subclass__() takes no arguments (2 given)",
)

# One unknown keyword names it; several report the count instead
# (`ArgErrUnknownKwds`, argument.py:620-627).
expect_type_error(
    "one_keyword",
    lambda: object.__init_subclass__(x=1),
    "object.__init_subclass__() got an unexpected keyword argument 'x'",
    "object.__init_subclass__() takes no keyword arguments",
)
expect_type_error(
    "two_keywords",
    lambda: object.__init_subclass__(x=1, y=2),
    "object.__init_subclass__() got 2 unexpected keyword arguments",
    "object.__init_subclass__() takes no keyword arguments",
)
# `cls` is the parameter's own name, and it is still not accepted by keyword.
expect_type_error(
    "cls_by_keyword",
    lambda: object.__init_subclass__(cls=1),
    "object.__init_subclass__() got an unexpected keyword argument 'cls'",
    "object.__init_subclass__() takes no keyword arguments",
)

# Keywords are collected before the positional overflow is judged, so a call
# carrying both reports the keyword.
expect_type_error(
    "keyword_beats_positional",
    lambda: object.__init_subclass__(1, x=1),
    "object.__init_subclass__() got an unexpected keyword argument 'x'",
    "object.__init_subclass__() takes no keyword arguments",
)

# Reached through a subclass, the message still names `object` — the function
# is named by where it is defined, not by the class the classmethod bound.
expect_type_error(
    "subclass_receiver",
    lambda: int.__init_subclass__(x=1),
    "object.__init_subclass__() got an unexpected keyword argument 'x'",
    "int.__init_subclass__() takes no keyword arguments",
)


# A class statement's keywords reach the same default and are refused there.
# The bodies run through `exec` in a bare namespace so that cpython's message,
# which names `__qualname__`, reads `A` rather than a `<locals>` path.
expect_type_error(
    "class_statement_keyword",
    lambda: exec("class A(x=1): pass", {}),
    "object.__init_subclass__() got an unexpected keyword argument 'x'",
    "A.__init_subclass__() takes no keyword arguments",
)
expect_type_error(
    "class_statement_two_keywords",
    lambda: exec("class A(x=1, y=2): pass", {}),
    "object.__init_subclass__() got 2 unexpected keyword arguments",
    "A.__init_subclass__() takes no keyword arguments",
)

# `type.__new__` forwards the class-definition keywords to the same place.
expect_type_error(
    "type_new_keyword",
    lambda: type.__new__(type, "A", (), {}, x=1),
    "object.__init_subclass__() got an unexpected keyword argument 'x'",
    "A.__init_subclass__() takes no keyword arguments",
)
expect_type_error(
    "type_call_keyword",
    lambda: type("A", (), {}, x=1),
    "object.__init_subclass__() got an unexpected keyword argument 'x'",
    "A.__init_subclass__() takes no keyword arguments",
)


# A class that defines its own `__init_subclass__` accepts what it declares,
# so none of the above applies to it.
class Accepting:
    seen = None

    def __init_subclass__(cls, /, tag=None, **kwargs):
        super().__init_subclass__(**kwargs)
        Accepting.seen = tag


class Tagged(Accepting, tag="here"):
    pass


assert Accepting.seen == "here", Accepting.seen

# `descr___subclasshook__(space, __args__)` (objectobject.py:136) declares
# `__args__` instead, so the same parser accepts anything and the hook answers
# `NotImplemented`.  The contrast is what shows the refusals above follow the
# declared signature rather than a blanket rule for `object`'s classmethods.
if IS_CPYTHON:
    expect_type_error(
        "subclasshook_positional",
        lambda: object.__subclasshook__(1, 2),
        None,
        "object.__subclasshook__() takes exactly one argument (2 given)",
    )
    expect_type_error(
        "subclasshook_keyword",
        lambda: object.__subclasshook__(x=1),
        None,
        "object.__subclasshook__() takes no keyword arguments",
    )
else:
    assert object.__subclasshook__(1, 2) is NotImplemented
    assert object.__subclasshook__(x=1) is NotImplemented

print("OK")
