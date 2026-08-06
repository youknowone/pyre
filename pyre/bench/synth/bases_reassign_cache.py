# No `max-pypy-ratio`: this fixture compiles no loop -- its jitstats record
# `loops_compiled=0` -- so a pypy ratio compares two interpreters' startup
# rather than any generated code, and reads whatever the host's process
# spawn cost happens to be that run. The jitstats baselines gate it.
# Reassigning type.__bases__ recomputes the MRO of the type and every
# subclass and invalidates the method/attribute lookup cache, so a method or
# class attribute that was resolved before the swap resolves through the new
# base afterwards.
def main():
    class A:
        def who(self):
            return "A"

    class B:
        def who(self):
            return "B"

    class C(A):
        pass

    class CC(C):
        pass

    c = C()
    cc = CC()
    print(c.who(), cc.who())  # prime the cache on the old hierarchy
    C.__bases__ = (B,)
    print(c.who(), cc.who())
    print([t.__name__ for t in C.__mro__])
    print([t.__name__ for t in CC.__mro__])

    # data attribute follows the swap too
    class D:
        x = 1

    class E:
        x = 2

    class F(D):
        pass

    f = F()
    print(f.x)
    F.__bases__ = (E,)
    print(f.x)

    # a method only defined on the new base becomes reachable
    class G:
        pass

    class H:
        def hello(self):
            return "H"

    class I(G):
        pass

    i = I()
    I.__bases__ = (H,)
    print(i.hello())


main()
