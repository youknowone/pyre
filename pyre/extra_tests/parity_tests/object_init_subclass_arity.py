"""`object.__init_subclass__` refuses what its declared signature refuses.

`descr___init_subclass__(space, w_cls)` (objectobject.py:139) declares one
parameter and no `__args__`, so everything past the bound class is refused by
the argument parser before the body runs — and the parser's `ArgErr` shapes
(argument.py:529-627) are what word the refusal.  Two consequences the ad-hoc
"takes no keyword arguments" spelling does not have: an extra *positional*
argument is an error too, and the message names the defining type, so it reads
`object.__init_subclass__()` even when the call arrives through a subclass or
through a class statement's keywords.

The runtimes word these refusals differently and disagree on which class the
message names, so what is asserted is the part they share — the method's own
`__init_subclass__()` and, where both go on to describe the count, `takes `.
Nothing is keyed off which interpreter is running.  Because the message is
never compared in full, the class statements below can be written directly
instead of through `exec` in a bare namespace, which earlier existed only to
keep one runtime's `__qualname__` out of a literal.
"""


def expect_type_error(label, fn, shared):
    """Assert `fn` is refused with a TypeError whose message contains `shared`."""
    try:
        result = fn()
    except TypeError as exc:
        assert shared in str(exc), (label, shared, str(exc))
    except BaseException as exc:
        raise AssertionError((label, type(exc).__name__, str(exc)))
    else:
        raise AssertionError((label, "no TypeError", result))


# The bound class alone is the whole accepted signature.
assert object.__init_subclass__() is None
assert int.__init_subclass__() is None


# An extra positional argument is refused.  pyre used to return None here,
# which is the row that makes this more than a wording difference.  Both
# refusals go on to describe the count, so `takes ` is shared as well.
expect_type_error(
    "one_extra_positional",
    lambda: object.__init_subclass__(1),
    "object.__init_subclass__() takes ",
)
expect_type_error(
    "two_extra_positional",
    lambda: object.__init_subclass__(1, 2),
    "object.__init_subclass__() takes ",
)

# Keywords are refused whatever they are named, including the parameter's own
# name, and one is described differently from several.
expect_type_error(
    "one_keyword", lambda: object.__init_subclass__(x=1), "object.__init_subclass__() "
)
expect_type_error(
    "two_keywords", lambda: object.__init_subclass__(x=1, y=2), "object.__init_subclass__() "
)
expect_type_error(
    "cls_by_keyword", lambda: object.__init_subclass__(cls=1), "object.__init_subclass__() "
)

# Keywords are collected before the positional overflow is judged, so a call
# carrying both is still refused.
expect_type_error(
    "keyword_beats_positional",
    lambda: object.__init_subclass__(1, x=1),
    "object.__init_subclass__() ",
)

# Reached through a subclass the call is refused the same way; which class the
# message names is where the runtimes part, so only the method is shared.
expect_type_error("subclass_receiver", lambda: int.__init_subclass__(x=1), "__init_subclass__() ")


# A class statement's keywords reach the same default and are refused there.
def class_statement_keyword():
    class A(x=1):
        pass


def class_statement_two_keywords():
    class A(x=1, y=2):
        pass


expect_type_error("class_statement_keyword", class_statement_keyword, "__init_subclass__() ")
expect_type_error(
    "class_statement_two_keywords", class_statement_two_keywords, "__init_subclass__() "
)

# `type.__new__` forwards the class-definition keywords to the same place.
expect_type_error(
    "type_new_keyword",
    lambda: type.__new__(type, "A", (), {}, x=1),
    "__init_subclass__() ",
)
expect_type_error(
    "type_call_keyword", lambda: type("A", (), {}, x=1), "__init_subclass__() "
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

# `descr___subclasshook__(space, __args__)` (objectobject.py:136) is the other
# classmethod on the same class, and it answers rather than refuses.  Only the
# one-argument call is asserted: how many arguments it accepts is the declared
# signature, which the runtimes do not share, but every one of them answers
# `NotImplemented` for the call the protocol actually makes.
assert object.__subclasshook__(int) is NotImplemented

print("OK")
