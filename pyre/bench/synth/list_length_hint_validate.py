# list(iterable) and list.extend(iterable) consult __length_hint__ before
# iterating (listobject.py _extend_from_iterable -> space.length_hint), which
# validates it: a negative hint raises ValueError, one exceeding a C ssize_t
# raises OverflowError.  A valid hint is used transparently.  Only the
# exception type is checked (the OverflowError message is install-specific).
# Output verified against CPython/PyPy.
N = 5000


def make(hint):
    class E:
        def __iter__(self):
            return iter((1, 2, 3))

        def __length_hint__(self):
            return hint

    return E()


def classify_list(hint):
    try:
        list(make(hint))
        return "ok"
    except ValueError:
        return "ValueError"
    except OverflowError:
        return "OverflowError"


def classify_extend(hint):
    try:
        r = []
        r.extend(make(hint))
        return "ok"
    except ValueError:
        return "ValueError"
    except OverflowError:
        return "OverflowError"


def main():
    hits = 0
    for _ in range(N):
        ok = (
            classify_list(-1) == "ValueError"
            and classify_list(2 ** 63) == "OverflowError"
            and classify_list(5) == "ok"
            and classify_extend(-1) == "ValueError"
            and classify_extend(2 ** 63) == "OverflowError"
            and classify_extend(10) == "ok"
        )
        if ok:
            hits += 1
    print(hits)


main()
