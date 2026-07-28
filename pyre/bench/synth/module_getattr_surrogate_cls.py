# module.py Module.descr_getattribute for a lone-surrogate attribute name: a
# module miss consults the module dict's PEP 562 __getattr__ first, then the
# class-level __getattr__ on the module's type (descroperation.py:242-245),
# just like an identifier name.  Results are compared, never printed, so the
# surrogate never reaches the stdout codec.

import types


class M(types.ModuleType):
    def __getattr__(self, name):
        return ("cls", name)


SUR = "\udc80"  # lone surrogate -> non-identifier attribute name


def main():
    m = M("m")
    # class-level __getattr__ reached for both an identifier and a surrogate name
    print("ident", getattr(m, "missing") == ("cls", "missing"))
    print("surr", getattr(m, SUR) == ("cls", SUR))

    # a module-dict PEP 562 hook wins over the class-level one, for both kinds
    def modhook(name):
        return ("dict", name)

    m.__getattr__ = modhook
    print("ident2", getattr(m, "again") == ("dict", "again"))
    print("surr2", getattr(m, SUR) == ("dict", SUR))

    # a module-dict hook that itself raises AttributeError also falls through
    # to the class-level __getattr__
    def raising(name):
        raise AttributeError("hook raised")

    m2 = M("m2")
    m2.__getattr__ = raising
    print("ident3", getattr(m2, "x") == ("cls", "x"))
    print("surr3", getattr(m2, SUR) == ("cls", SUR))

    # no hook anywhere: AttributeError for both name kinds
    bare = types.ModuleType("bare")
    for label, nm in (("ident4", "gone"), ("surr4", SUR)):
        try:
            getattr(bare, nm)
            print(label, "NO-RAISE")
        except AttributeError:
            print(label, "AttributeError")

    # a raising module-dict hook with NO class-level fallback propagates
    class Plain(types.ModuleType):
        pass

    p = Plain("p")
    p.__getattr__ = raising
    for label, nm in (("ident5", "gone"), ("surr5", SUR)):
        try:
            getattr(p, nm)
            print(label, "NO-RAISE")
        except AttributeError:
            print(label, "AttributeError")

    # a hot loop over the routed surrogate access stays stable once compiled
    ok = 0
    for _ in range(20000):
        if getattr(m, SUR) == ("dict", SUR):
            ok += 1
    print("hot", ok)


main()
