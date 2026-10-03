# pyre-check: max-pypy-ratio=27.3
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
# dynasm 12.9x, cranelift 13.6x; the ceiling is twice the slower,
# rounded up to one decimal place. n is sized so pypy's execution-only
# time clears the floor gate's bar. The mutating body now inlines:
# FORMAT_WITH_SPEC is not a call boundary, and `foriter_dirty_bound` admits
# the `Dirty` store once the callee has a seeded frame, which is the resume
# `perform_call` records.  `loops_compiled` is 1.  The two `str` additions
# inside `__format__` stay on the builtin slot.
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
