"""
RPython calc interpreter - equivalent to majit/examples/calc.
Translatable by RPython with JIT support.

Usage:
  # Translate with JIT:
  PYPY_DONT_RUN_SUBPROCESS=1 pypy rpython/bin/rpython --opt=jit majit/bench/rpython_calc.py
  # Run:
  ./rpython_calc-c [n]
"""
from __future__ import print_function
import os
import sys
import time

from rpython.rlib.jit import JitDriver


# --- bytecodes (flat encoding: [opcode, arg, opcode, arg, ...]) ---
LOAD_CONST    = 0
LOAD_VAR      = 1
STORE_VAR     = 2
ADD           = 3
SUB           = 4
MUL           = 5
DIV           = 6
MOD           = 7
LT            = 8
LE            = 9
EQ            = 10
NE            = 11
GT            = 12
GE            = 13
JUMP_IF_FALSE = 14
JUMP          = 15
PRINT         = 16
HALT          = 17


def make_sum_program(n):
    return [
        LOAD_CONST, 0,         # 0: sum = 0
        STORE_VAR, 0,          # 2
        LOAD_CONST, 0,         # 4: i = 0
        STORE_VAR, 1,          # 6
        LOAD_VAR, 1,           # 8: loop header
        LOAD_CONST, n,         # 10
        LT, 0,                 # 12: i < n
        JUMP_IF_FALSE, 34,     # 14: goto end (instruction 17 * 2)
        LOAD_VAR, 0,           # 16: sum + i
        LOAD_VAR, 1,           # 18
        ADD, 0,                # 20
        STORE_VAR, 0,          # 22
        LOAD_VAR, 1,           # 24: i + 1
        LOAD_CONST, 1,         # 26
        ADD, 0,                # 28
        STORE_VAR, 1,          # 30
        JUMP, 8,               # 32: goto loop
        LOAD_VAR, 0,           # 34: result
        HALT, 0,               # 36
    ]


# --- JIT driver ---
def get_printable_location(pc, bytecode):
    return "pc=%d" % pc

jitdriver = JitDriver(
    greens=['pc', 'bytecode'],
    reds=['sp', 'stack', 'variables'],
    get_printable_location=get_printable_location,
)


# --- interpreter ---
def interp(bytecode):
    pc = 0
    stack = [0] * 256
    sp = 0
    variables = [0] * 26

    while True:
        jitdriver.jit_merge_point(
            pc=pc, bytecode=bytecode, sp=sp, stack=stack, variables=variables
        )

        opcode = bytecode[pc]
        arg = bytecode[pc + 1]
        pc += 2

        if opcode == LOAD_CONST:
            stack[sp] = arg
            sp += 1
        elif opcode == LOAD_VAR:
            stack[sp] = variables[arg]
            sp += 1
        elif opcode == STORE_VAR:
            sp -= 1
            variables[arg] = stack[sp]
        elif opcode == ADD:
            sp -= 1
            b = stack[sp]
            sp -= 1
            a = stack[sp]
            stack[sp] = a + b
            sp += 1
        elif opcode == SUB:
            sp -= 1
            b = stack[sp]
            sp -= 1
            a = stack[sp]
            stack[sp] = a - b
            sp += 1
        elif opcode == MUL:
            sp -= 1
            b = stack[sp]
            sp -= 1
            a = stack[sp]
            stack[sp] = a * b
            sp += 1
        elif opcode == DIV:
            sp -= 1
            b = stack[sp]
            sp -= 1
            a = stack[sp]
            if b != 0:
                stack[sp] = a // b
            sp += 1
        elif opcode == MOD:
            sp -= 1
            b = stack[sp]
            sp -= 1
            a = stack[sp]
            if b != 0:
                stack[sp] = a % b
            sp += 1
        elif opcode == LT:
            sp -= 1
            b = stack[sp]
            sp -= 1
            a = stack[sp]
            if a < b:
                stack[sp] = 1
            else:
                stack[sp] = 0
            sp += 1
        elif opcode == LE:
            sp -= 1
            b = stack[sp]
            sp -= 1
            a = stack[sp]
            if a <= b:
                stack[sp] = 1
            else:
                stack[sp] = 0
            sp += 1
        elif opcode == EQ:
            sp -= 1
            b = stack[sp]
            sp -= 1
            a = stack[sp]
            if a == b:
                stack[sp] = 1
            else:
                stack[sp] = 0
            sp += 1
        elif opcode == NE:
            sp -= 1
            b = stack[sp]
            sp -= 1
            a = stack[sp]
            if a != b:
                stack[sp] = 1
            else:
                stack[sp] = 0
            sp += 1
        elif opcode == GT:
            sp -= 1
            b = stack[sp]
            sp -= 1
            a = stack[sp]
            if a > b:
                stack[sp] = 1
            else:
                stack[sp] = 0
            sp += 1
        elif opcode == GE:
            sp -= 1
            b = stack[sp]
            sp -= 1
            a = stack[sp]
            if a >= b:
                stack[sp] = 1
            else:
                stack[sp] = 0
            sp += 1
        elif opcode == JUMP_IF_FALSE:
            sp -= 1
            cond = stack[sp]
            if not cond:
                pc = arg
        elif opcode == JUMP:
            pc = arg
            jitdriver.can_enter_jit(
                pc=pc, bytecode=bytecode, sp=sp, stack=stack, variables=variables
            )
        elif opcode == PRINT:
            sp -= 1
            os.write(1, str(stack[sp]) + '\n')
        elif opcode == HALT:
            sp -= 1
            return stack[sp]
        else:
            raise RuntimeError("unknown opcode")
    return 0


# --- entry point ---
def entry_point(argv):
    n = 10000000
    if len(argv) > 1:
        n = int(argv[1])

    prog = make_sum_program(n)

    # warmup
    interp(make_sum_program(100))

    t0 = time.time()
    result = interp(prog)
    t1 = time.time()

    os.write(1, "sum(0..%d) = %d\n" % (n, result))
    elapsed_ms = int((t1 - t0) * 1000)
    os.write(1, "time = %d ms\n" % elapsed_ms)
    return 0


def target(*args):
    return entry_point, None


if __name__ == '__main__':
    entry_point(sys.argv)
