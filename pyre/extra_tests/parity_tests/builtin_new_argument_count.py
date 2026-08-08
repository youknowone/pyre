"""`type.__new__`, `bool.__new__` and `complex.__new__` decide on the argument count.

`typeobject.py:886-911 descr__new__` receives the metatype as its own gateway
parameter and the rest as `__args__`, so the one-versus-three form is settled by
`len(__args__.arguments_w)` before any argument is read.  `_precheck_for_new`
(`typeobject.py:1001-1003`) then refuses a non-type metatype, and the metaclass
that survives `_calculate_metaclass` still has to pass
`W_TypeObject.check_user_subclass` (`typeobject.py:555-567`) on the way through
`allocate_instance`.

`complexobject.py:325-327 descr__new__` is a gateway of
`(space, w_complextype, w_real, w_imag=None)` with
`@unwrap_spec(w_real=WrappedDefault(0.0))`, so one to three positionals counting
the class are accepted and the surplus is refused before any subtype `__init__`
sees it.  The class then goes through the same `tp_new_wrapper` check every
builtin `__new__` uses, which is what keeps a complex payload from being stamped
with a type of a different layout.

The wordings differ between implementations, so most of the file asserts only
the exception type; each refusal below is one the reference raises too.  The
block at the end goes further and asserts the part of the message every runtime
words alike, as a containment.  Neither half is keyed off which interpreter is
running — a fixture that compares one literal on one implementation and another
literal on another has stopped comparing them.

The one place a runtime is named is the one-argument form of `type.__new__`,
which the reference dropped altogether: there the outcome differs, not its
spelling, and no single assertion covers both.
"""

import sys


def raises_type_error(label, fn):
    try:
        result = fn()
    except TypeError:
        return
    raise AssertionError(f"{label} returned {result!r} instead of raising TypeError")


def message(fn):
    try:
        fn()
    except TypeError as exc:
        return str(exc)
    raise AssertionError("expected TypeError")


# A non-type metatype is refused, not used to build a class.
raises_type_error("type.__new__(42, ...)", lambda: type.__new__(42, "A", (), {}))
raises_type_error("type.__new__(None, ...)", lambda: type.__new__(None, "A", (), {}))
raises_type_error("type.__new__('s', ...)", lambda: type.__new__("s", "A", (), {}))

# A type that is not a subtype of `type` is refused after metaclass calculation.
raises_type_error("type.__new__(int, ...)", lambda: type.__new__(int, "A", (), {}))
raises_type_error("type.__new__(str, ...)", lambda: type.__new__(str, "A", (), {}))

# Neither one nor three arguments behind the metatype.
raises_type_error("type.__new__()", lambda: type.__new__())
raises_type_error("type.__new__(type)", lambda: type.__new__(type))
raises_type_error("type.__new__(42, 'A', ())", lambda: type.__new__(42, "A", ()))
raises_type_error("type.__new__(type, 'A', ())", lambda: type.__new__(type, "A", ()))

# Three arguments, so the count is settled and the name is what gets reported.
raises_type_error("type(1, (), {})", lambda: type(1, (), {}))
raises_type_error("type('A', [], {})", lambda: type("A", [], {}))
raises_type_error("type('A', (), [])", lambda: type("A", (), []))

class Meta(type):
    pass


# The one-argument form belongs to `type` alone; every other metatype names only
# the three-argument one.  The reference dropped the form altogether and refuses
# both, so the accepting half is asserted off it.
raises_type_error("type.__new__(Meta, 1)", lambda: type.__new__(Meta, 1))
if sys.implementation.name != "cpython":
    assert type.__new__(type, 1) is int
    assert type.__new__(type, "s") is str

# The three-argument form still builds a class, through `type` and a metaclass.
built = type("Built", (), {"x": 1})
assert built.__name__ == "Built", built.__name__
assert built.x == 1
assert type(built) is type

metabuilt = Meta("MetaBuilt", (), {})
assert type(metabuilt) is Meta


class ViaSuper(type):
    def __new__(mcls, name, bases, namespace):
        return super().__new__(mcls, name, bases, namespace)


class UsesSuper(metaclass=ViaSuper):
    pass


assert type(UsesSuper) is ViaSuper
assert UsesSuper.__name__ == "UsesSuper"

# Heap types the interpreter builds for itself go through the same entry.
import ast  # noqa: E402

assert isinstance(ast.parse("1"), ast.Module)

# `bool.__new__` counts the class argument along with the value.
raises_type_error("bool(1, 2)", lambda: bool(1, 2))
raises_type_error("bool(1, 2, 3)", lambda: bool(1, 2, 3))
raises_type_error("bool(x=1)", lambda: bool(x=1))
assert bool() is False
assert bool(1) is True

# `complex.__new__` counts the class argument along with `real` and `imag`.
raises_type_error("complex(1, 2, 3)", lambda: complex(1, 2, 3))
raises_type_error("complex(1, 2, 3, 4)", lambda: complex(1, 2, 3, 4))
raises_type_error("complex.__new__(complex, 1, 2, 3)", lambda: complex.__new__(complex, 1, 2, 3))
assert complex() == 0j
assert complex(1) == 1 + 0j
assert complex(1, 2) == 1 + 2j
assert complex(imag=2) == 2j


class ComplexInit(complex):
    def __init__(self, *args, **kwargs):
        pass


# The count is settled in `__new__`, so an `__init__` that would swallow the
# surplus never gets the chance.
raises_type_error("ComplexInit(1, 2, 3)", lambda: ComplexInit(1, 2, 3))
assert ComplexInit(1, 2) == 1 + 2j
assert type(ComplexInit(1, 2)) is ComplexInit

# A class argument that is not a type, or not a subtype of `complex`, is refused
# rather than stamped onto the fresh complex.
raises_type_error("complex.__new__(1)", lambda: complex.__new__(1))
raises_type_error("complex.__new__(int)", lambda: complex.__new__(int))
raises_type_error("complex.__new__(object)", lambda: complex.__new__(object))
assert complex.__new__(complex, 1, 2) == 1 + 2j
assert type(complex.__new__(ComplexInit, 1, 2)) is ComplexInit

# What the refusals say, to the extent the runtimes say the same thing.
# `descr__new__` names both accepted counts for `type` itself and only the
# three-argument form for any other metatype, and `_precheck_for_new` runs after
# that decision, so an unvalidated metatype reaches `%N`; a runtime that checks
# the metatype first answers about the metatype instead.  Each row below asserts
# the part they have in common rather than one message per implementation —
# comparing a different literal on each runtime is not a parity assertion.
assert "type.__new__() takes " in message(lambda: type.__new__(type))
assert "type.__new__() takes " in message(lambda: type.__new__(type, "A", ()))
# The metatype's own name reaches the count, so a named metaclass names itself
# where `type` is named for it.
assert ".__new__() takes exactly 3 arguments (1 given)" in message(lambda: type.__new__(Meta, 1))
assert "X is not a type object (int)" in message(lambda: type.__new__(42, "A", (), {}))
assert "type.__new__(int): int is not a subtype of type" in message(
    lambda: type.__new__(int, "A", (), {})
)
assert "argument 1 must be str" in message(lambda: type(1, (), {}))
assert "argument 2 must be tuple, not list" in message(lambda: type("A", [], {}))
assert "argument 3 must be dict, not list" in message(lambda: type("A", (), []))
assert "keyword argument" in message(lambda: bool(x=1))

# The subtype refusal is the one every runtime words alike, so the whole
# sentence is what they share.
assert "complex.__new__(int): int is not a subtype of complex" in message(
    lambda: complex.__new__(int)
)
assert "complex.__new__(object): object is not a subtype of complex" in message(
    lambda: complex.__new__(object)
)
# A class argument that is not a type at all reaches the same sentence for every
# builtin `__new__`.  What differs is whether the call it came from is named
# first, and whether the rejected type's name is quoted.
assert "X is not a type object (" in message(lambda: complex.__new__(1))
assert "X is not a type object (" in message(lambda: float.__new__(1))

print("OK")
