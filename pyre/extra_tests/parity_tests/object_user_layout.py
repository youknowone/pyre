# CPython-suite gap: no CPython test checks that exact object() and a
# user-class instance share attribute, pickle, copy, and weakref behavior
# while living on different interp-level layouts.
# parity-tests reason: typedef.py _getusercls(W_ObjectObject) /
# W_ObjectObjectUserDictWeakrefable versus W_ObjectObject for object().

"""Exact object() and user-class instances.

`object()` rejects attributes and `__class__` assignment. A plain class,
a slotted class, and a subclass keep mapdict attributes, slots, pickle,
copy, weakref, and `__del__`. The hot loop is what makes the JIT record
the allocation.
"""

import copy
import gc
import pickle
import weakref


class A:
    pass


class B:
    __slots__ = ("a",)


class C(A):
    pass


def err(fn):
    try:
        fn()
        return "ok"
    except Exception as e:
        return type(e).__name__


o = object()
a = A()
a.x = 1
print("get", a.x)
del a.x
print("del", err(lambda: a.x), hasattr(a, "x"))
a.x = 2
print("dict", sorted(a.__dict__.items()))
b = B()
b.a = 3
print("slot", b.a, err(lambda: b.__dict__), err(lambda: setattr(b, "z", 1)))
print("types", type(o).__name__, type(a).__name__, type(b).__name__, type(C()).__name__)
print(
    "isinstance",
    isinstance(o, object),
    isinstance(a, object),
    isinstance(a, A),
    isinstance(C(), A),
    isinstance(o, A),
)
a2 = A()
a2.x = 4
a2.__class__ = C
print("reclass", type(a2).__name__, a2.x, isinstance(a2, A), isinstance(a2, C))
a2.__class__ = A
print("reclass2", type(a2).__name__, a2.x)
print("obj-attr", err(lambda: setattr(o, "x", 1)))
print("obj-class", err(lambda: setattr(o, "__class__", A)))
print("eq", o == o, o == object(), a == a, a == A())
print("hash", type(hash(o)).__name__, hash(a) == hash(a))


def rshape(s):
    head, sep, tail = s.partition(" object at ")
    return (head, bool(sep), tail.startswith("0x"), tail.endswith(">"))


print("repr", rshape(repr(o)), rshape(repr(a)), rshape(repr(b)))
pa = pickle.loads(pickle.dumps(a))
print("pickle", type(pa).__name__, pa.x)
pb = pickle.loads(pickle.dumps(b))
print("pickle-slot", type(pb).__name__, pb.a)
ca = copy.copy(a)
print("copy", type(ca).__name__, ca.x, ca is a)
da = copy.deepcopy(a)
print("deepcopy", type(da).__name__, da.x, da is a)
alive = A()
ref = weakref.ref(alive)
print("weak", ref() is alive)
print("weak-obj", err(lambda: weakref.ref(object())))
seen = []


class D:
    def __del__(self):
        seen.append("d")


d = D()
del d
gc.collect()
print("del", seen)


def hot(n):
    total = 0
    i = 0
    while i < n:
        x = A()
        x.k = i
        total += x.k
        i += 1
    return total


print("hot", hot(2000))
print("OK")
