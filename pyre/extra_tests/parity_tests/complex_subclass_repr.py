class MadComplex(complex):
    def __repr__(self):
        return f"{self.imag:g}j{self.real:+g}"


value = MadComplex(-3, 4)
assert repr(value) == "4j-3"


class Number(complex):
    def __repr__(self):
        if self.imag == 0.0:
            return f"{self.real:g}"
        if self.real == 0.0:
            return f"{self.imag:g}j"
        return f"({self.real:g}+{self.imag:g}j)"

    __str__ = __repr__


number = Number(3.14)
assert repr(number) == "3.14"
assert str(number) == "3.14"


class BadRepr(complex):
    def __repr__(self):
        return 1


try:
    repr(BadRepr())
except TypeError as error:
    assert "__repr__ returned non-string" in str(error)
else:
    raise AssertionError("non-string __repr__ result was accepted")
