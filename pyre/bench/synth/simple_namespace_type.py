# `types.SimpleNamespace` is a dedicated type: keyword construction into the
# instance dict, a `namespace(...)` repr, `__dict__` exposure, structural
# equality, unhashability, and `type(sys.implementation)` identity. Multi-key
# repr ordering differs between implementations, so only single-key/empty repr
# and order-independent behaviour are printed. Output verified against
# CPython/PyPy.
from types import SimpleNamespace


def main():
    n = SimpleNamespace(a=1, b="x")
    n.c = 3
    print(n.a, n.b, n.c)
    print(repr(SimpleNamespace(only=42)))
    print(repr(SimpleNamespace()))
    print(n.__dict__ == {"a": 1, "b": "x", "c": 3})
    print(SimpleNamespace(a=1, b=2) == SimpleNamespace(b=2, a=1))
    print(SimpleNamespace(a=1) == SimpleNamespace(a=2))
    print(SimpleNamespace(a=1) != SimpleNamespace(a=2))
    print(SimpleNamespace(a=1) == 5)
    print(SimpleNamespace(a=1) != 5)
    try:
        hash(SimpleNamespace())
        print("HASHABLE")
    except TypeError:
        print("unhashable")
    import sys
    print(type(n) is type(sys.implementation))
    print(type(n).__name__, type(n).__qualname__, type(n).__module__)

    class Sub(SimpleNamespace):
        pass

    s = Sub(z=9)
    print(s.z, isinstance(s, SimpleNamespace))
    r = SimpleNamespace()
    r.self = r
    print(repr(r))

    # construction updates __dict__ directly, so a subclass __setattr__ is not
    # invoked while building; a later assignment does go through it.
    log = []

    class Logged(SimpleNamespace):
        def __setattr__(self, key, value):
            log.append(key)
            object.__setattr__(self, key, value)

    lg = Logged(a=1, b=2)
    print(log, lg.a, lg.b)
    lg.c = 3
    print(log)


main()
