test_super_list = super(list)
assert test_super_list.__self__ is None
assert test_super_list.__self_class__ is None
assert test_super_list.__thisclass__ == list


class testA:
    a = 1


class testB(testA):
    b = 1


superB = super(testB)
assert superB.__thisclass__ == testB
assert superB.__self_class__ is None
assert superB.__self__ is None


# CPython 3.14 LOAD_SUPER_ATTR calls the value loaded from the `super`
# global. Bit 1 of the opcode chooses the zero- or two-argument call shape;
# exercise both repeatedly so the generated JIT residual keeps that contract.
class _SuperResult:
    value = 41


_super_calls = []


def _shadowed_super(*args):
    _super_calls.append(args)
    return _SuperResult()


super = _shadowed_super


class _Shadowed:
    def zero(self):
        return super().value

    def two(self):
        return super(_Shadowed, self).value


_shadowed = _Shadowed()
for _ in range(100):
    assert _shadowed.zero() == 41
    assert _shadowed.two() == 41

assert _super_calls[0] == ()
assert _super_calls[1] == (_Shadowed, _shadowed)


# test_super.py:test_various___class___pathologies — an explicit class-body
# binding belongs to the namespace, while methods still close over the
# implicit cell that type.__new__ fills with the new class.
class _ExplicitClass:
    def captured_class(self):
        return __class__

    __class__ = 413


_explicit_class = _ExplicitClass()
assert _explicit_class.captured_class() is _ExplicitClass
assert _explicit_class.__class__ == 413


# A class-body read still uses LOAD_NAME even when a nested method makes
# `__class__` a cell variable.  It therefore sees the surrounding global,
# while the method observes the class installed into the distinct cell.
__class__ = type


class _ReadClassName:
    namespace_value = __class__

    def captured_class(self):
        return __class__


assert _ReadClassName.namespace_value is type
assert _ReadClassName().captured_class() is _ReadClassName
del __class__


# The pinned compiler likewise spells CPython's DELETE_NAME as DELETE_DEREF
# when the implicit class cell exists.  Deleting the absent namespace binding
# raises NameError and must not clear that cell instead.
try:
    class _DeleteClassName:
        def captured_class(self):
            return __class__

        del __class__
except NameError:
    pass
else:
    raise AssertionError("NameError expected")


# test_super.py:test___class___mro — type.__new__ fills __classcell__ before a
# custom metaclass mro() runs, so code invoked by mro() already observes the
# newly allocated class.
_class_seen_during_mro = None


class _MroMeta(type):
    def mro(self):
        self.__dict__["capture_class"]()
        return type.mro(self)


class _ClassCellDuringMro(metaclass=_MroMeta):
    def capture_class():
        global _class_seen_during_mro
        _class_seen_during_mro = __class__


assert _class_seen_during_mro is _ClassCellDuringMro
