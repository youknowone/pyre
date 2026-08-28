# CPython-suite gap: pickletools tests cannot execute opcode walking in a bridge.
# parity-tests reason: this guards pyre JIT bridge abort result propagation.

"""A bridge walk that adopts a blackhole terminal must keep the frame's result.

``pickletools.optimize`` ends in a single ``return out.getvalue()``. Under the
JIT, the guard-failure bridge walk over its opcode loop can stop inside the
reconstructed callee, adopt the blackhole chain it has already driven to a
return, and then abort the trace. The adopted terminal carries that frame's
``DoneWithThisFrame`` value, which is the function's return value; dropping it
leaves the caller with neither a result nor a replay, so ``optimize`` returns
``None`` and ``pickle.loads`` raises ``EOFError: Ran out of input``.

Two rounds over the varying protocol/width matrix are required: the trace is
compiled during the first round and the guard fails in the second, and a fixed
input never reaches the failing shape.
"""

import pickle
import pickletools
import sys

for _round in range(2):
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        n = sys.maxsize
        while n:
            for expected in (-n, n):
                data = pickletools.optimize(pickle.dumps(expected, proto))
                assert data is not None, (
                    f"optimize() returned None (proto={proto} n={expected})"
                )
                got = pickle.loads(data)
                assert got == expected, (
                    f"round-trip gave {got!r}, want {expected!r} "
                    f"(proto={proto} n={expected})"
                )
            n = n >> 1

print("OK")
