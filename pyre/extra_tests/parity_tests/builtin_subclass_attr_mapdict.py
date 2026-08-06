import gc
import os


class T(tuple):
    x = "class"


class I(int):
    x = "class"


class S(str):
    x = "class"


class DataDescriptor:
    def __get__(self, obj, owner):
        return "data"

    def __set__(self, obj, value):
        pass


class TD(tuple):
    d = DataDescriptor()


class ID(int):
    d = DataDescriptor()


class SD(str):
    d = DataDescriptor()


shadowed = (T((1, 2)), I(5), S("hi"))
for obj in shadowed:
    obj.x = "instance"
assert tuple(obj.x for obj in shadowed) == ("instance", "instance", "instance")

described = (TD((1,)), ID(5), SD("hi"))
for obj in described:
    obj.__dict__["d"] = "instance"
assert tuple(obj.d for obj in described) == ("data", "data", "data")

print(tuple(obj.x for obj in shadowed))
print(tuple(obj.d for obj in described))


def load_mark(values):
    total = 0
    for value in values:
        total += value.mark
    return total


def store_mark(values):
    for value in values:
        value.mark = 19


int_user = I(5)
str_user = S("user")
tuple_user = T((3, 4))
for value in (int_user, str_user, tuple_user):
    value.mark = 7

exact_values = (5, "interned_mapdict_exact", (1, 2))
for user, exact in zip((int_user, str_user, tuple_user), exact_values):
    try:
        load_mark([user] * 4000 + [exact])
    except AttributeError:
        pass
    else:
        raise AssertionError(type(user).__name__)
    try:
        store_mark([user] * 4000 + [exact])
    except AttributeError:
        pass
    else:
        raise AssertionError(type(user).__name__)
    assert user.mark == 19


class Before(str):
    pass


class After(str):
    @property
    def chosen(self):
        return "descriptor"


swapped = Before("value")
swapped.__dict__["chosen"] = "instance"
swapped.__class__ = After
assert swapped.chosen == "descriptor"


class SlottedStr(str):
    __slots__ = ("slot", "__dict__")


slotted = SlottedStr("value")
slotted.slot = "slot-value"
slotted.extra = "dict-value"
assert (slotted.slot, slotted.extra) == ("slot-value", "dict-value")

stat_value = os.stat_result((0,) * 10)
assert stat_value.st_atime_ns is None
assert stat_value.st_mtime_ns is None

for value in (int_user, str_user, tuple_user):
    assert any(candidate is value for candidate in gc.get_objects())
    assert any(candidate is value for candidate in gc.get_referents([value]))

print("OK")
