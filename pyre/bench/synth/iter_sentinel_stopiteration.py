# When the callable of iter(callable, sentinel) raises StopIteration, the
# iterator latches exhausted and surfaces a bare StopIteration; the callable's
# StopIteration value does not leak to the consumer.
def main():
    def boom():
        raise StopIteration("leaked message")

    it = iter(boom, 42)
    try:
        next(it)
        print("no error")
    except StopIteration as e:
        print("args:", e.args)

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
