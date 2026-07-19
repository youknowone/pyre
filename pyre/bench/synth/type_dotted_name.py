# type.__name__ / __qualname__ dot handling: a heap type built with a dotted
# name string (type('a.b', (), {})) keeps the name verbatim on both getters,
# while a static/builtin dotted tp_name strips to its final component. An
# explicit __qualname__ in the namespace wins, and nested-class qualnames are
# preserved.
import re


def main():
    T = type('mod.Name', (), {})
    print(T.__name__, T.__qualname__)

    U = type('a.b', (), {'__qualname__': 'Outer.Inner'})
    print(U.__name__, U.__qualname__)

    class Plain:
        pass

    print(Plain.__name__, Plain.__qualname__)

    class Outer:
        class Inner:
            pass

    print(Outer.Inner.__name__, Outer.Inner.__qualname__)

    pat = type(re.compile(''))
    print(pat.__name__, pat.__qualname__)
    print(int.__name__, int.__qualname__)


main()
