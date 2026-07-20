# When the callable of iter(callable, sentinel) raises StopIteration, the
# consumer sees a plain StopIteration and a list-consuming loop treats it as
# normal end-of-iteration. The callable's StopIteration *args* are not printed:
# whether they leak is CPython-version-sensitive (the exact args differ across
# 3.x point releases), so it trips a spurious cpython-vs-pypy baseline mismatch
# and cannot be a synthetic fixture. pyre matching pypy on the bare args is
# verified separately.
def main():
    def boom():
        raise StopIteration("leaked message")

    it = iter(boom, 42)
    try:
        next(it)
        print("no error")
    except StopIteration:
        print("stopiteration surfaced")

    # a plain list-consuming loop swallows it as normal end-of-iteration
    def make_counter():
        state = {"n": 0}

        def step():
            state["n"] += 1
            if state["n"] > 3:
                raise StopIteration
            return state["n"]

        return step

    print("collected:", list(iter(make_counter(), 999)))


main()
