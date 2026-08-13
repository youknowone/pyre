# A `with` block in a hot FOR_ITER body, whose handler runs every tenth
# iteration. The whole-frame FOR_ITER gate declines LOAD_SPECIAL, so this frame
# runs interpreted; the answer is recorded here for the day it is admitted.
# `exception_with_exit_self_null_slot` is the same shape written as a `while`
# loop, where no gate stands between it and the JIT.
N = 20000


class Context:
    def __init__(self):
        self.exits = 0

    def __enter__(self):
        return 3

    def __exit__(self, exc_type, exc_value, traceback):
        self.exits += 1
        return exc_type is ValueError


def main():
    context = Context()
    total = 0
    for i in range(N):
        with context as value:
            total += value
            if i % 10 == 0:
                raise ValueError
            total += i
    print(total, context.exits)


main()
# Expected: 180060000 20000
