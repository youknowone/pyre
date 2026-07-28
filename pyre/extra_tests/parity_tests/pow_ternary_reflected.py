events = []


class Base:
    def __pow__(self, other, modulus=None):
        events.append(("pow", type(other).__name__, type(modulus).__name__))
        return NotImplemented


class Exponent:
    def __rpow__(self, other, modulus=None):
        events.append(("rpow", type(other).__name__, type(modulus).__name__))
        return "reflected"


class Modulus:
    def __pow__(self, other, modulus=None):
        raise AssertionError("the modulus must not receive __pow__")

    def __rpow__(self, other, modulus=None):
        raise AssertionError("the modulus must not receive __rpow__")


assert pow(Base(), Exponent(), Modulus()) == "reflected"
assert events == [
    ("pow", "Exponent", "Modulus"),
    ("rpow", "Base", "Modulus"),
]


class IntExponent(int):
    def __rpow__(self, other, modulus=None):
        return IntExponent(pow(int(other), int(self), int(modulus)))


result = pow(2, IntExponent(3), 5)
assert result == 3
assert type(result) is IntExponent

# A modulus subclass does not take part in Python-level reflected dispatch.
result = pow(2, 3, IntExponent(5))
assert result == 3
assert type(result) is int


subtype_events = []


class Parent:
    def __pow__(self, other, modulus=None):
        subtype_events.append("parent pow")
        return "parent"

    def __rpow__(self, other, modulus=None):
        subtype_events.append("parent rpow")
        return "inherited"


class Child(Parent):
    def __rpow__(self, other, modulus=None):
        subtype_events.append("child rpow")
        return NotImplemented


assert pow(Parent(), Child(), 7) == "parent"
assert subtype_events == ["child rpow", "parent pow"]
