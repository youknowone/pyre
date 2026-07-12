# Regression for wasm CA frames: this self-recursive body reaches the raw
# float-power residual call (CallF, hence the host jit_call trampoline) and
# allocates a string at every recursive level. Once the recursion bridge is
# hot, CA must be declined: a moving nursery callee frame cannot survive the
# trampoline retaining its pre-call frame pointer.


def descend(n):
    if n < 2:
        return n
    allocated = str(n) * 512
    powered = (n + 1.25) ** 1.5
    # Preserve the ordinary fib recursion shape while retaining `powered` in
    # the trace. `int(powered)` allocates a boxed result on each level.
    return len(allocated) + descend(n - 1) + descend(n - 2) + int(powered) - int(powered)


def run():
    total = 0
    i = 0
    while i < 240:
        total += descend(10)
        i += 1
    return total


print(run())
