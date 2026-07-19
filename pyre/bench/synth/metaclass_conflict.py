# An unrelated-metaclass clash among the bases (or an explicit metaclass that
# is not a subclass of a base's metaclass) is a hard TypeError, not silently
# resolved to the first metaclass.
class M1(type):
    pass


class M2(type):
    pass


class M3(M1):
    pass


class A(metaclass=M1):
    pass


class B(metaclass=M2):
    pass


class D(metaclass=M3):
    pass


def main():
    # class-statement inference: A(M1) + B(M2) clash
    try:
        class C(A, B):
            pass
        print("class C(A,B): no error")
    except TypeError as e:
        print("class C(A,B):", e)

    # explicit metaclass that clashes with a base's metaclass
    try:
        class E(A, metaclass=M2):
            pass
        print("class E(A,meta=M2): no error")
    except TypeError as e:
        print("class E(A,meta=M2):", e)

    # 3-arg type() inference clash
    try:
        type("F", (A, B), {})
        print("type(F,(A,B)): no error")
    except TypeError as e:
        print("type(F,(A,B)):", e)

    # more-derived metaclass wins with no clash (M1 + M3 subclass -> M3)
    class G(A, D):
        pass
    print("G meta:", type(G).__name__)


main()
