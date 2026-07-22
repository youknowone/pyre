from enum import Enum, IntEnum, IntFlag, auto


class Color(Enum):
    RED = 1
    BLUE = 2


class Number(IntEnum):
    ONE = 1
    TWO = 2


class Permission(IntFlag):
    READ = auto()
    WRITE = auto()


assert repr(Color) == "<enum 'Color'>"

try:
    delattr(Color, "RED")
except AttributeError:
    pass
else:
    raise AssertionError("Enum members must not be deletable")

assert Color(1) is Color.RED
assert list(Color) == [Color.RED, Color.BLUE]
assert Number.ONE == 1
assert Permission.READ | Permission.WRITE == 3
assert ~Permission.READ is Permission.WRITE


class BaseNumber(IntEnum):
    @property
    def one(self):
        return self.name


class DerivedNumber(BaseNumber):
    one = auto()
    two = auto()


# EnumType installs a redirect descriptor and stores the member on it.  The
# mapdict value must remain the IntEnum object, not be unboxed to a plain int.
assert DerivedNumber.one is DerivedNumber._member_map_["one"]
assert DerivedNumber.one.__class__ is DerivedNumber


class FloatSubclass(float):
    pass


class Holder:
    pass


holder = Holder()
holder.value = FloatSubclass(1.5)
assert holder.value.__class__ is FloatSubclass
holder_with_unboxed_slot = Holder()
holder_with_unboxed_slot.value = 2.0
holder_with_unboxed_slot.value = FloatSubclass(2.5)
assert holder_with_unboxed_slot.value.__class__ is FloatSubclass


class Meta(type):
    def __repr__(cls):
        return f"<custom {cls.__name__}>"


class Custom(metaclass=Meta):
    pass


assert repr(Custom) == "<custom Custom>"

# Type metadata is an object-valued getset, not a freshly boxed string on
# every access.  `inspect.classify_class_attrs` relies on this identity when
# locating `__name__` and `__qualname__` on the defining class.
assert Custom.__name__ is Custom.__name__
assert Custom.__qualname__ is Custom.__qualname__


class Name(str):
    pass


custom_name = Name("Renamed")
Custom.__name__ = custom_name
assert Custom.__name__ is custom_name


# `__slots__` accepts any iterable.  A `__doc__` slot suppresses the default
# class-level None entry, while another slot iterable still gets that default.
DocSlotSet = type("DocSlotSet", (), {"__slots__": {"__doc__"}})
assert type(DocSlotSet.__dict__["__doc__"]).__name__ == "member_descriptor"

doc_slots = iter(["__doc__"])
DocSlotIterator = type("DocSlotIterator", (), {"__slots__": doc_slots})
assert DocSlotIterator.__slots__ is doc_slots
assert list(doc_slots) == []
assert type(DocSlotIterator.__dict__["__doc__"]).__name__ == "member_descriptor"

PlainSlotSet = type("PlainSlotSet", (), {"__slots__": {"value"}})
assert PlainSlotSet.__dict__["__doc__"] is None


print("stdlib enum ok")
