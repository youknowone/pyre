# pyre-check: max-pypy-ratio=165
# `FORMAT_WITH_SPEC` whose `__format__` is a Python method, inside a `for`
# body. `foriter_format_with_spec` only formats ints, so the residual never
# enters a user frame there; the body scan admits the opcode on the grounds
# that a user `__format__` runs Python exactly as a user `__str__` does under
# `CONVERT_VALUE`, and this is what exercises that.
#
# `__format__` mutates a pre-existing object (`self.calls`), so the counter is
# the exactly-once assertion: a dropped iteration lowers it and a replayed one
# raises it. `spec` is threaded into the result so a wrong spec operand shows
# up in the output rather than only in the count.
# Output verified against CPython/PyPy.
#
# The ceiling is far above `foriter_format_with_spec`'s because the residual
# calls a Python `__format__` every iteration and the trace does not inline it,
# while pypy does: n is sized so pypy's execution-only time clears the floor
# the harness clamps to, without which no ratio would be applied at all. The
# reading is 60-68x/81-89x, and the ceiling is twice the slowest; that is
# the same band the other user-dunder-per-iteration fixtures sit in
# (`getattribute_override_no_bind` 374, `property_protocol_hot` 364).
N = 500000


class Tagged:
    def __init__(self):
        self.calls = 0

    def __format__(self, spec):
        self.calls += 1
        return "<" + spec + ">"


def main():
    t = Tagged()
    total = 0
    last = ""
    for _ in range(N):
        last = f"{t:>{4}}"
        total += len(last)
    print(total, t.calls, last, f"{t:x}", t.calls)


main()
