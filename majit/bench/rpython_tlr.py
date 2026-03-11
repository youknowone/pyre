"""
RPython TLR interpreter — equivalent to majit/examples/tlr.
Direct port of rpython/jit/tl/tlr.py with benchmarking.

Usage:
  PYPY_DONT_RUN_SUBPROCESS=1 pypy rpython/bin/rpython --batch --opt=jit majit/bench/rpython_tlr.py
  ./rpython_tlr-c [n]
"""
from __future__ import print_function
import os
import sys
import time

from rpython.rlib.jit import JitDriver


MOV_A_R    = 1
MOV_R_A    = 2
JUMP_IF_A  = 3
SET_A      = 4
ADD_R_TO_A = 5
RETURN_A   = 6
ALLOCATE   = 7
NEG_A      = 8


def get_printable_location(pc, bytecode):
    return "pc=%d" % pc

tlrjitdriver = JitDriver(
    greens=['pc', 'bytecode'],
    reds=['a', 'regs'],
    get_printable_location=get_printable_location,
)


def interpret(bytecode, a):
    regs = []
    pc = 0
    while True:
        tlrjitdriver.jit_merge_point(bytecode=bytecode, pc=pc, a=a, regs=regs)
        opcode = ord(bytecode[pc])
        pc += 1
        if opcode == MOV_A_R:
            n = ord(bytecode[pc])
            pc += 1
            regs[n] = a
        elif opcode == MOV_R_A:
            n = ord(bytecode[pc])
            pc += 1
            a = regs[n]
        elif opcode == JUMP_IF_A:
            target = ord(bytecode[pc])
            pc += 1
            if a:
                if target < pc:
                    tlrjitdriver.can_enter_jit(bytecode=bytecode, pc=target,
                                               a=a, regs=regs)
                pc = target
        elif opcode == SET_A:
            # Extended: support negative and large values via 4-byte encoding
            b0 = ord(bytecode[pc])
            b1 = ord(bytecode[pc + 1])
            b2 = ord(bytecode[pc + 2])
            b3 = ord(bytecode[pc + 3])
            a = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
            if a >= 0x80000000:
                a -= 0x100000000
            pc += 4
        elif opcode == ADD_R_TO_A:
            n = ord(bytecode[pc])
            pc += 1
            a += regs[n]
        elif opcode == RETURN_A:
            return a
        elif opcode == ALLOCATE:
            n = ord(bytecode[pc])
            pc += 1
            regs = [0] * n
        elif opcode == NEG_A:
            a = -a


def encode_i32(val):
    """Encode a signed 32-bit integer as 4 bytes, little-endian."""
    if val < 0:
        val += 0x100000000
    return [val & 0xff, (val >> 8) & 0xff, (val >> 16) & 0xff, (val >> 24) & 0xff]


def make_sum_program(n):
    """Build a sum(0..n) program for TLR.

    regs: 0=counter, 1=i, 2=sum, 3=const_1
    """
    code = []
    # ALLOCATE 4
    code += [ALLOCATE, 4]
    # SET_A n; MOV_A_R 0  (counter = n)
    code += [SET_A] + encode_i32(n) + [MOV_A_R, 0]
    # SET_A 0; MOV_A_R 1  (i = 0)
    code += [SET_A] + encode_i32(0) + [MOV_A_R, 1]
    # SET_A 0; MOV_A_R 2  (sum = 0)
    code += [SET_A] + encode_i32(0) + [MOV_A_R, 2]
    # SET_A 1; MOV_A_R 3  (const_1 = 1)
    code += [SET_A] + encode_i32(1) + [MOV_A_R, 3]
    # loop header (pc = 30)
    loop_pc = len(code)
    # MOV_R_A 2; ADD_R_TO_A 1; MOV_A_R 2  (sum += i)
    code += [MOV_R_A, 2, ADD_R_TO_A, 1, MOV_A_R, 2]
    # MOV_R_A 1; ADD_R_TO_A 3; MOV_A_R 1  (i += 1)
    code += [MOV_R_A, 1, ADD_R_TO_A, 3, MOV_A_R, 1]
    # SET_A -1; ADD_R_TO_A 0; MOV_A_R 0  (counter -= 1)
    code += [SET_A] + encode_i32(-1) + [ADD_R_TO_A, 0, MOV_A_R, 0]
    # JUMP_IF_A loop_pc  (if counter != 0)
    code += [JUMP_IF_A, loop_pc]
    # MOV_R_A 2; RETURN_A
    code += [MOV_R_A, 2, RETURN_A]
    return ''.join([chr(c & 0xff) for c in code])


# --- entry point ---
def entry_point(argv):
    n = 10000000
    if len(argv) > 1:
        n = int(argv[1])

    prog = make_sum_program(n)

    # warmup
    interpret(make_sum_program(100), 0)

    t0 = time.time()
    result = interpret(prog, 0)
    t1 = time.time()

    elapsed_ms = int((t1 - t0) * 1000)
    os.write(1, "sum(0..%d) = %d\n" % (n, result))
    os.write(1, "time = %d ms\n" % elapsed_ms)
    return 0


def target(*args):
    return entry_point, None


if __name__ == '__main__':
    entry_point(sys.argv)
