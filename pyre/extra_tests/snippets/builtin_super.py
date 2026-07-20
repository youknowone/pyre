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
