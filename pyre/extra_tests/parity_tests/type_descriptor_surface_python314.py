class Base:
    inherited = 1


class Child(Base):
    local = 2


assert {"local", "inherited"} <= set(type.__dir__(Child))
assert "__call__" not in type.__dir__(Child)
assert type.__getattribute__(Child, "local") == 2
assert type.__getattribute__(Child, "inherited") == 1

for method in (type.__dir__, lambda obj: type.__getattribute__(obj, "x")):
    try:
        method(1)
    except TypeError:
        pass
    else:
        raise AssertionError("type descriptor accepted a non-type receiver")

descriptor = type.__dict__["__abstractmethods__"]
assert not hasattr(Child, "__abstractmethods__")
descriptor.__set__(Child, frozenset({"missing"}))
assert Child.__abstractmethods__ == frozenset({"missing"})
try:
    Child()
except TypeError:
    pass
else:
    raise AssertionError("abstract class was instantiated")
descriptor.__delete__(Child)
assert not hasattr(Child, "__abstractmethods__")
assert isinstance(Child(), Child)


class Meta(type):
    @property
    def __abstractmethods__(cls):
        return "metaclass override"


class WithMeta(metaclass=Meta):
    pass


assert WithMeta.__abstractmethods__ == "metaclass override"
assert Meta.__abstractmethods__ is Meta.__dict__["__abstractmethods__"]
