# A `__slots__` entry naming a class variable raises ValueError at class
# creation, while a duplicate `__slots__` entry is silently ignored.
# Output verified against CPython/PyPy.
N = 5000


def make_conflict():
    class C:
        __slots__ = ("a",)
        a = 1

    return C


def make_duplicate():
    class D:
        __slots__ = ("x", "x")

    return D


def main():
    hits = 0
    for _ in range(N):
        try:
            make_conflict()
            conflict = "no-error"
        except ValueError:
            conflict = "ValueError"
        dup_ok = make_duplicate() is not None
        if conflict == "ValueError" and dup_ok:
            hits += 1
    print(hits)


main()
