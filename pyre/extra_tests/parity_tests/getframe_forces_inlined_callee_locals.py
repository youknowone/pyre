# CPython-suite gap: the frame tests never hand out the frame of a recursive
# callee whose surrounding loop the JIT has already compiled.
# parity-tests reason: this guards the locals a compiled trace is holding in
# registers for an inlined callee against a `sys._getframe` that forces it.

"""Handing out a running frame must materialize the locals that frame holds.

`sys._getframe(0)` gives application code the running frame, which forces the
virtualizable behind it.  The frame handed out here belongs to an inlined
recursive callee published on the execution context, not to the traced
virtualizable, and forcing has to write THAT frame's storage: the compiled
trace was holding `r` in a register, and a `LOAD_FAST` after the force reads
the frame's array.

An unwritten slot reads as a null, and `r != r` then raises
``TypeError: comparison on null operand`` -- an error no Python implementation
produces for a bound local.  The recursion and the ``n > 300`` arm are both
load-bearing: without the divergence `n` never exceeds 212, the arm never runs,
and the frame is never handed out from a compiled callee.
"""

import sys

MOD = 1000003
ROUNDS = 9000
ESCAPE_ABOVE = 300


def rec(n, seen):
    if n <= 1:
        return n
    if n % 2 == 0:
        r = (rec(n // 2, seen) * 3 + 7) % MOD
    else:
        r = (rec(n - 1, seen) + n * 5) % MOD
    if n > ESCAPE_ABOVE:
        seen[0] += 1
        frame = sys._getframe(0)
        # Touch the frame so the force cannot be elided, then read a local the
        # compiled trace was holding for this callee.
        if frame.f_lineno < 0:
            seen[1] += 1
        if r != r:
            seen[1] += 1
    return r


def main():
    seen = [0, 0]
    acc = 0
    for i in range(1, ROUNDS + 1):
        n = (i * 37) % 211 + 2
        if i > ROUNDS - 2000:
            n = n * 31 + 1
        acc = (acc + rec(n, seen)) % MOD
    return acc, seen


acc, seen = main()
results = []
if acc != 578965:
    print(f"acc: got {acc!r}, want 578965")
    results.append(False)
if seen[0] == 0:
    print("the escaping arm never ran; the fixture measures nothing")
    results.append(False)
if seen[1] != 0:
    print(f"frame reads disagreed {seen[1]} time(s)")
    results.append(False)
if not results:
    print("OK")
