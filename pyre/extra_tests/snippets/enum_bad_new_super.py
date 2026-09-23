# pyre-check: gate=1
# lib-python/3/test/test_enum.py TestStrEnumClass.test_bad_new_super.
# `super().__new__` during StrEnum class creation must raise TypeError,
# not SIGSEGV.  CPython 3.14 compiles this as LOAD_ATTR with the self-bit
# (`__new__ + NULL|self`); pyre binds through compute_load_method_bound.
# Super has a custom getattribute (descriptor.py W_Super.getattribute),
# so callmethod.py:46 has_object_getattribute is false and the call site
# must not prepend a receiver.

from enum import StrEnum

raised = None
try:

    class BadSuper(StrEnum):
        def __new__(cls, value):
            obj = super().__new__(cls, value)
            return obj

        failed = 1
except TypeError as exc:
    raised = str(exc)

assert raised is not None, "expected TypeError from super().__new__ during StrEnum creation"
assert "do not use" in raised and "__new__" in raised, raised
