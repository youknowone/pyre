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
