SENTINEL = object()


class E(Exception):
    calls = 0
    getattr_calls = 0

    def __getattribute__(self, name):
        type(self).calls += 1
        if name == "marker":
            return SENTINEL
        if name == "args":
            return ("overridden-args",)
        if name == "__traceback__":
            return "overridden-traceback"
        return super().__getattribute__(name)

    def __getattr__(self, name):
        type(self).getattr_calls += 1
        if name == "fallback":
            return "fallback-value"
        raise AttributeError(name)


e = E("real-args")
start = E.calls
assert e.marker is SENTINEL
assert E.calls == start + 1

start = E.calls
for _ in range(4000):
    assert e.marker is SENTINEL
assert E.calls == start + 4000

assert e.args == ("overridden-args",)
assert e.__traceback__ == "overridden-traceback"
assert e.fallback == "fallback-value"
assert E.getattr_calls == 1
before = E.getattr_calls
assert e.marker is SENTINEL
assert E.getattr_calls == before


class L(list):
    calls = 0

    def __getattribute__(self, name):
        type(self).calls += 1
        if name == "marker":
            return SENTINEL
        return super().__getattribute__(name)


list_subclass = L([1, 2, 3])
assert list_subclass.marker is SENTINEL
assert L.calls == 1

try:
    raise E("raised")
except E as raised:
    assert raised.marker is SENTINEL
    assert raised.args == ("overridden-args",)
    assert raised.__traceback__ == "overridden-traceback"


# Receivers whose builtin type carries a `__getattribute__` of its own read
# their attributes unchanged: super proxies, bound methods, unions, generic
# aliases, weak proxies and struct objects.
class Q:
    attr = "q-attr"

    def f(self):
        return "q-f"


q = Q()
bound = q.f
assert bound.__func__ is Q.f
assert bound.__self__ is q
assert bound() == "q-f"

assert (int | str).__args__ == (int, str)
assert list[int].__origin__ is list

import struct
import weakref

assert struct.Struct("i").size == struct.calcsize("i")
proxy = weakref.proxy(q)
assert proxy.attr == "q-attr"


class Base:
    def who(self):
        return "base"


class Derived(Base):
    def who(self):
        return "derived:" + super().who()


assert Derived().who() == "derived:base"

print("OK")
