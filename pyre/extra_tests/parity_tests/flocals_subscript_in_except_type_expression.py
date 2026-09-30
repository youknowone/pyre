# CPython-suite gap: the suite never evaluates an `except` type expression that
# reads `f_locals` inside a loop the tracer is recording.
# parity-tests reason: the subscript is traced through the proxy's own
# `__getitem__` while the exception the `except` clause tests is still pending,
# so the descent must hand that exception back to the match that follows.

"""An ``except`` clause whose type is read through ``f_locals``.

``except sys._getframe().f_locals["E"]`` evaluates the subscript after the
exception is raised and before it is matched. Every iteration must still catch
the ``KeyError`` and count it.
"""

import sys

ROUNDS = 4000


def caught(rounds):
    E = KeyError
    hits = 0
    for i in range(rounds):
        try:
            {}[i]
        except sys._getframe().f_locals["E"]:
            hits += 1
    return hits


assert caught(ROUNDS) == ROUNDS, caught(ROUNDS)
print("OK")
