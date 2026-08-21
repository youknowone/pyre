# pyre-check: selfcheck
# The BINDING half of `wrapper_subclass_get_override`.  That fixture's
# `overridden(*args)` absorbs a wrongly prepended leading argument, so it can
# only ever see a wrong UNWRAP; this one's callable takes exactly one
# parameter, so a wrongly prepended class raises instead of being swallowed.
#
# `callmethod.py` LOAD_METHOD pushes a `null_or_self` alongside the attribute
# that `getattr` already resolved.  For a plain `classmethod` the two halves
# pair up: the attribute lookup hands back the raw function instead of
# allocating a bound method, and the binder answers "prepend the class".  A
# `classmethod` SUBCLASS that overrides `__get__` resolves through the override
# instead, so the attribute is whatever the override returned and there is
# nothing for the class to be prepended to.  A binder that tests the wrapper by
# layout rather than exactly still answers "prepend the class", and the call
# arrives with one argument too many.
#
# One loop per receiver shape that reaches the binder — instance, type, and the
# two builtin-storage payloads — plus two `getattr` loops as the control: they
# resolve the same attribute through the same descriptor without going near the
# binder, so they stay right whatever the binder does.  Every loop must report
# the OVERRIDE.
#
# Measured before the fix: the first loop died with `TypeError: overridden()
# takes 1 positional argument but 2 were given`, and identically with
# `PYRE_NO_JIT=1` — an interpreter defect, not a JIT one.  CPython 3.14.2 and
# PyPy 3.11.13 agree on every line.
#
# Self-checking rather than output-comparing so the expectation sits next to
# the loop it belongs to.  Each check reads the value the loop left BEHIND it:
# an assert inside the loop body would stop that loop compiling and the fixture
# would then gate the interpreter it was written to leave.
import sys

N = 20000


class GetOverridingClassMethod(classmethod):
    def __get__(self, obj, cls=None):
        return overridden


def overridden(x):
    return ('override', x)


def wrapped_class(cls, x):
    return ('wrapped', x)


class Attrs:
    cm = GetOverridingClassMethod(wrapped_class)


class ListAttrs(list):
    cm = GetOverridingClassMethod(wrapped_class)


class TupleAttrs(tuple):
    cm = GetOverridingClassMethod(wrapped_class)


def on_type():
    seen = None
    i = 0
    while i < N:
        seen = Attrs.cm(i)
        i += 1
    return seen


def on_instance():
    obj = Attrs()
    seen = None
    i = 0
    while i < N:
        seen = obj.cm(i)
        i += 1
    return seen


def on_builtin_storage_type():
    seen = None
    i = 0
    while i < N:
        seen = ListAttrs.cm(i)
        i += 1
    return seen


def on_list_payload():
    obj = ListAttrs()
    seen = None
    i = 0
    while i < N:
        seen = obj.cm(i)
        i += 1
    return seen


def on_tuple_payload():
    obj = TupleAttrs()
    seen = None
    i = 0
    while i < N:
        seen = obj.cm(i)
        i += 1
    return seen


def getattr_on_instance():
    obj = Attrs()
    seen = None
    i = 0
    while i < N:
        seen = getattr(obj, 'cm')(i)
        i += 1
    return seen


def getattr_on_type():
    seen = None
    i = 0
    while i < N:
        seen = getattr(Attrs, 'cm')(i)
        i += 1
    return seen


def main():
    want = ('override', N - 1)
    cases = [
        ('type receiver', on_type),
        ('instance receiver', on_instance),
        ('builtin-storage type receiver', on_builtin_storage_type),
        ('list payload receiver', on_list_payload),
        ('tuple payload receiver', on_tuple_payload),
        ('getattr on instance (control)', getattr_on_instance),
        ('getattr on type (control)', getattr_on_type),
    ]

    failures = []
    for label, fn in cases:
        try:
            got = fn()
        except TypeError as exc:
            # The defect's own signature: the class arrived as an extra
            # leading argument.  Report it as a value so a different TypeError
            # cannot be mistaken for this one.
            failures.append(f"{label}: raised TypeError({exc})")
            continue
        if got != want:
            failures.append(f"{label}: got {got!r}, want {want!r}")

    if failures:
        for line in failures:
            print("FAIL", line)
        return 1
    print("PASS classmethod subclass takes no self-binding")
    return 0


sys.exit(main())
