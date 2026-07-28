# pyre-check: max-pypy-ratio=30
# Regression guard: an INLINED callee that inspects its own frame via
# sys._getframe(0) must still see ITS OWN frame, not its caller's. The walker
# executes residuals concretely while an inline push never runs the
# interpreter's call sequence, so ec.topframeref still names the caller unless
# the inline declines or materialises the callee's frame. Publishing a frame
# for only the inline levels that have one shifts every _getframe up a level
# instead of fixing it, and the shift is invisible to a JIT-vs-JIT comparison
# -- both pyre paths move together. Hence the assert against the literal
# expected chain rather than a self-consistency check.
import sys


def leaf(x):
    frame = sys._getframe(0)
    return (frame.f_code.co_name, frame.f_back.f_code.co_name)


def main():
    seen = set()
    total = 0
    for i in range(20000):
        seen.add(leaf(i))
        total += i
    print(total, sorted(seen))


main()
