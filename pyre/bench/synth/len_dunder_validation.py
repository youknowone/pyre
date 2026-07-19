# `len()` runs the `__len__` result through `__index__` coercion and the
# length checks: negative -> ValueError, non-int -> TypeError, and a value
# too large for a machine word -> OverflowError.
def check(cls):
    try:
        print(len(cls()))
    except (ValueError, TypeError, OverflowError) as e:
        print(type(e).__name__, e)


def main():
    class Neg:
        def __len__(self):
            return -1

    class Float:
        def __len__(self):
            return 1.5

    class Big:
        def __len__(self):
            return 2 ** 100

    class Idx:
        def __len__(self):
            class I:
                def __index__(self):
                    return 3
            return I()

    for cls in (Neg, Float, Big, Idx):
        check(cls)


main()
