"""Type base relationships, heap flags, mutability, and baseability parity."""

import sys
import types


class A:
    pass


class B(A):
    pass


assert object.__base__ is None
assert type.__base__ is object
assert A.__base__ is object
assert B.__base__ is A
assert isinstance(A.__flags__, int)
assert A.__flags__ & (1 << 9)

# ExceptionGroup is the mutable heap-type exception-group leaf; BaseExceptionGroup
# remains an immutable static type.
ExceptionGroup._pyre_type_flags_probe = 1
assert ExceptionGroup._pyre_type_flags_probe == 1
del ExceptionGroup._pyre_type_flags_probe

for non_base in (slice, types.MappingProxyType):
    try:
        type("BadSubclass", (non_base,), {})
    except TypeError:
        pass
    else:
        raise AssertionError(f"{non_base.__name__} must not be subclassable")

# Both are read-only and non-deletable, but the two interpreters word the
# AttributeError differently, so only its type is asserted.
for cls in (A, B):
    for name in ("__flags__", "__base__"):
        original = getattr(cls, name)
        try:
            setattr(cls, name, None)
        except AttributeError:
            pass
        else:
            raise AssertionError(f"{cls.__name__}.{name} must be read-only")
        assert getattr(cls, name) == original

        try:
            delattr(cls, name)
        except AttributeError:
            pass
        else:
            raise AssertionError(f"{cls.__name__}.{name} must not be deletable")
        assert getattr(cls, name) == original

print("OK")
