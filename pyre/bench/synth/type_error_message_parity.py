# Error-message parity across a few type/builtin operations that both
# reference implementations word identically.
def main():
    class A:
        pass

    # __class__ must be a class
    try:
        A().__class__ = 5
    except TypeError as e:
        print(e)

    # empty tuple to __bases__
    class C(A):
        pass
    try:
        C.__bases__ = ()
    except TypeError as e:
        print(e)

    # incompatible solid base names the current best base ('A'), not 'C'
    class B(int):
        pass
    try:
        C.__bases__ = (B,)
    except TypeError as e:
        print(e)

    # next() on a non-iterator names the type
    for x in (5, [1, 2]):
        try:
            next(x)
        except TypeError as e:
            print(e)

    # complex.__format__ rejects non-float presentation codes as 'complex'
    for spec in ("d", "x", "s", "%"):
        try:
            format(1 + 2j, spec)
        except ValueError as e:
            print(e)


main()
