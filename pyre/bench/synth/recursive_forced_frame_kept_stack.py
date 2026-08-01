# A self-recursive callee whose frame is forced, with a kept operand-stack slot
# beneath an inlined comprehension.
#
# `locals()` forces the callee PyFrame, so a guard fired inside the callee
# rebuilds it through the materialized-frame recipe arm rather than the virtual
# one. The `f(n - 1) +` result sits on the operand stack across the
# comprehension, and `pyframe.py:396-403 popvalue_maybe_none` clears every slot
# the compare pops -- so the frame image at that guard is already past the
# clears while the published `valuestackdepth` is still the opcode-start depth.
# Reconstructing the operands from that image seeds NULL and the re-executed
# body dereferences it: SIGSEGV, not a wrong answer.
def f(n):
    d = locals()
    if n < 2:
        return n
    return f(n - 1) + len([j for j in range(3)]) + f(n - 2)


print(f(16))
