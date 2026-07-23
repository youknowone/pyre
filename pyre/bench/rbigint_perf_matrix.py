"""Focused steady-state workloads for PyPy rbigint ↔ pyre RBigInt parity.

Usage: rbigint_perf_matrix.py CASE [ROUNDS]
Each case keeps operand sizes stable and mutates at least one operand so a
meta-tracer cannot fold the arbitrary-precision operation out of the loop.
"""

import math
import sys


CHECK_MASK = (1 << 127) - 1


def add_sub(rounds):
    a = (1 << 4095) + (1 << 2047) + 0x123456789ABCDEF
    b = (1 << 4011) + (1 << 997) + 0xFEDCBA987654321
    x = a
    checksum = 0
    for i in range(rounds):
        x = x + b
        x = x - a
        x ^= i
        checksum ^= x & CHECK_MASK
    return checksum


def mul_school(rounds):
    x = (1 << 700) + (1 << 353) + 0x123456789
    y = (1 << 637) + (1 << 211) + 0xABCDEF
    keep = (1 << 756) - 1
    checksum = 0
    for i in range(rounds):
        product = x * y
        checksum ^= product & CHECK_MASK
        x = ((product >> 601) & keep) | (1 << 700) | (i & 255)
    return checksum


def mul_karatsuba(rounds):
    x = (1 << 4095) + (1 << 2111) + 0x123456789
    y = (1 << 4031) + (1 << 1901) + 0xABCDEF
    keep = (1 << 4158) - 1
    checksum = 0
    for i in range(rounds):
        product = x * y
        checksum ^= product & CHECK_MASK
        x = ((product >> 3973) & keep) | (1 << 4095) | (i & 255)
    return checksum


def mul_lopsided(rounds):
    x = (1 << 16383) + (1 << 10001) + 0x123456789
    y = (1 << 2047) + (1 << 1031) + 0xABCDEF
    keep = (1 << 16447) - 1
    checksum = 0
    for i in range(rounds):
        product = x * y
        checksum ^= product & CHECK_MASK
        x = ((product >> 1981) & keep) | (1 << 16383) | (i & 255)
    return checksum


def div_classic(rounds):
    numerator = (1 << 4095) + (1 << 3001) + 0x123456789
    divisor = (1 << 755) + (1 << 311) + 0xABCDEF
    checksum = 0
    for i in range(rounds):
        q, r = divmod(numerator + checksum + i, divisor)
        checksum ^= (q ^ r) & CHECK_MASK
    return checksum


def div_bz(rounds):
    numerator = (1 << 12287) + (1 << 9011) + 0x123456789
    divisor = (1 << 4095) + (1 << 2011) + 0xABCDEF
    checksum = 0
    for i in range(rounds):
        q, r = divmod(numerator + checksum + i, divisor)
        checksum ^= (q ^ r) & CHECK_MASK
    return checksum


def pow_mod(rounds):
    modulus = (1 << 4096) - (1 << 211) - 189
    x = (1 << 4001) + (1 << 1999) + 0x123456789
    checksum = 0
    for i in range(rounds):
        x = pow(x ^ i, 65537, modulus)
        checksum ^= x & CHECK_MASK
    return checksum


def bitwise(rounds):
    a = -((1 << 8191) + (1 << 4001) + 0x123456789)
    b = (1 << 8063) + (1 << 3011) + 0xABCDEF
    x = a
    checksum = 0
    for i in range(rounds):
        x = ((x & b) ^ (~a | i)) ^ b
        checksum ^= x & CHECK_MASK
    return checksum


def shifts(rounds):
    x = -((1 << 4095) + (1 << 2001) + 0x123456789)
    for i in range(rounds):
        shift = i & 255
        # Keep the operand width stable and feed both shift results into the
        # next iteration.  Mask only once at the end: doing a 4096-bit `&`
        # here would benchmark a third arbitrary-precision operation.
        x = ((x << shift) >> shift) ^ i
    return x & CHECK_MASK


def decimal_format(rounds):
    x = 10 ** 1999 + 10 ** 997 + 123456789
    checksum = 0
    for i in range(rounds):
        text = str(x + i)
        checksum ^= len(text) + ord(text[-1])
    return checksum


def decimal_parse(rounds):
    prefix = "9876543210" * 200
    checksum = 0
    for i in range(rounds):
        x = int(prefix + str(i % 10))
        checksum ^= (x & CHECK_MASK) + x.bit_length()
    return checksum


def byte_roundtrip(rounds):
    size = 1024
    x = (1 << (size * 8 - 1)) + (1 << 4001) + 0x123456789
    checksum = 0
    for i in range(rounds):
        data = (x ^ i).to_bytes(size, "little")
        y = int.from_bytes(data, "little")
        checksum ^= y & CHECK_MASK
    return checksum


def gcd_case(rounds):
    a = ((1 << 4095) + (1 << 2011) + 0x123456789) * 65537
    b = ((1 << 4001) + (1 << 1997) + 0xABCDEF) * 65537
    checksum = 0
    for i in range(rounds):
        checksum ^= math.gcd(a + i * 65537, b) & CHECK_MASK
    return checksum


def isqrt_case(rounds):
    x = (1 << 8191) + (1 << 4011) + 0x123456789
    checksum = 0
    for i in range(rounds):
        checksum ^= math.isqrt(x + i) & CHECK_MASK
    return checksum


CASES = {
    "add_sub": (add_sub, 200000),
    "mul_school": (mul_school, 20000),
    "mul_karatsuba": (mul_karatsuba, 2000),
    "mul_lopsided": (mul_lopsided, 800),
    "div_classic": (div_classic, 6000),
    "div_bz": (div_bz, 1200),
    "pow_mod": (pow_mod, 200),
    "bitwise": (bitwise, 100000),
    "shifts": (shifts, 100000),
    "decimal_format": (decimal_format, 1500),
    "decimal_parse": (decimal_parse, 1500),
    "byte_roundtrip": (byte_roundtrip, 6000),
    "gcd": (gcd_case, 1000),
    "isqrt": (isqrt_case, 500),
}


def main():
    case = sys.argv[1]
    function, default_rounds = CASES[case]
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else default_rounds
    print(function(rounds))


main()
