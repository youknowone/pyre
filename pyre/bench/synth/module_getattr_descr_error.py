# module.py Module.descr_getattribute (PEP 562): an AttributeError raised by
# the normal module lookup — including a module-type data / non-data
# descriptor's __get__ (objspace.py:694-699) — is caught and routed to the
# module-level __getattr__, the same as a plain attribute miss.  Without a
# module __getattr__ the error propagates.  The descriptors live on a
# ModuleType subclass so the base module slot stays intact.

import types


class DataDesc:
    def __get__(self, obj, objtype=None):
        raise AttributeError("data descr")

    def __set__(self, obj, value):
        pass  # __set__ makes it a *data* descriptor


class NonDataDesc:
    def __get__(self, obj, objtype=None):
        raise AttributeError("non-data descr")


class M(types.ModuleType):
    ddesc = DataDesc()
    ndesc = NonDataDesc()


def with_hook():
    m = M("m")

    def modgetattr(attr):
        return "hook:" + attr

    m.__getattr__ = modgetattr
    return m


def main():
    m = with_hook()
    # a data descriptor whose __get__ raises AttributeError routes to the hook
    print("data", getattr(m, "ddesc"))
    # a non-data descriptor whose __get__ raises AttributeError routes too
    print("nondata", getattr(m, "ndesc"))
    # a plain miss routes to the hook as well
    print("miss", getattr(m, "nope"))

    # no module __getattr__: the descriptor AttributeError propagates unchanged
    bare = M("bare")
    try:
        getattr(bare, "ddesc")
        print("bare", "NO-RAISE")
    except AttributeError:
        print("bare", "AttributeError")

    # a hot loop keeps the routed result stable once the access is compiled
    seen = set()
    for _ in range(30000):
        seen.add(getattr(m, "ddesc"))
    print("hot", sorted(seen))


main()
