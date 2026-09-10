# pyre-check: gate=1
"""A compiled + must leave last_instr on the BINARY opcode, not 0.

sys._getframe().f_lineno reads last_instr.  A transparent helper walk
that published last_instr=0 made this report the function's first line.
"""

import sys


def add_loop(n):
    i = 0
    total = 0
    lineno = 0
    while i < n:
        total = total + i
        if i == n - 1:
            lineno = sys._getframe().f_lineno
        i = i + 1
    return total, lineno


def main():
    first = add_loop.__code__.co_firstlineno
    total, lineno = add_loop(3000)
    assert total == 4498500, total
    # last_instr=0 reports `def` / the first body line.  The getframe
    # call sits several opcodes into the loop.
    assert first + 5 <= lineno <= first + 8, (lineno, first)


main()
