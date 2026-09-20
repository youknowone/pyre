# pyre-check: gate=1
# A custom `__getattribute__` produced the attribute itself, so the method-call
# path must not infer a binding from the class's MRO on top of it:
# `_PyObject_GetMethod` skips the self-binding optimization whenever the type
# carries its own `tp_getattro`, and pushes NULL.  An override handing back the
# class's own raw function therefore calls it with the arguments written at the
# call site and nothing prepended.
#
# The receiver's storage must not change that answer.  A plain instance and a
# list-layout subclass reach two different arms of the same binding decision,
# and only the first one used to consult `__getattribute__`.


def check(cls):
    obj = cls()
    try:
        obj.f(1)
    except TypeError as exc:
        return str(exc)
    raise AssertionError(f"{cls.__name__}.f must not receive a bound self")


class Plain:
    def __getattribute__(self, name):
        if name == "f":
            return object.__getattribute__(type(self), "f")
        return object.__getattribute__(self, name)

    def f(self, x):
        return x


class ListBacked(list):
    def __getattribute__(self, name):
        if name == "f":
            return object.__getattribute__(type(self), "f")
        return object.__getattribute__(self, name)

    def f(self, x):
        return x


for klass in (Plain, ListBacked):
    message = check(klass)
    assert "missing 1 required positional argument" in message, (klass, message)
    assert "'x'" in message, (klass, message)


# The same override returning something that is NOT the class's own function
# takes no binding either, and stays callable on its own terms.
class Detached:
    def __getattribute__(self, name):
        if name == "f":
            return lambda x: ("free", x)
        return object.__getattribute__(self, name)

    def f(self, x):
        return ("method", x)


assert Detached().f(1) == ("free", 1)
