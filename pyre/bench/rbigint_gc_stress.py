"""GC and arithmetic stress for the RPython-compatible rbigint payload.

The state stays above the Karatsuba cutoff and is collected frequently, so a
missing RBigInt -> GcArray(Signed) edge is likely to become a crash or checksum
mismatch.  Keep the final line stable: check.py-style runners can compare it
across CPython, interpreter-only pyre, and JIT-enabled pyre.
"""

import gc
import sys


STATE_BITS = 4096
STATE_MASK = (1 << STATE_BITS) - 1
MODULUS = (1 << 4093) - 159
CHECK_MODULUS = (1 << 127) - 1


def run(rounds, collect_every):
    x = (1 << 4095) + (1 << 2057) + 0x123456789ABCDEF
    y = (1 << 4087) + (1 << 1999) + 0xFEDCBA987654321
    checksum = 0

    for i in range(rounds):
        # Multiplication remains in the multi-digit/Karatsuba range.  Masking
        # bounds the working set while still allocating fresh digit arrays.
        x = (x * (y | 1) + (x << (i % 29)) + i) & STATE_MASK
        y = (y * 1000003 + (x >> (i % 61)) + 17) & STATE_MASK

        signed_x = -x if i & 1 else x
        signed_y = -(y | 1) if i & 2 else (y | 1)
        quotient, remainder = divmod(signed_x, signed_y)

        modular = pow((x ^ y) + 3, 13 + (i % 5), MODULUS)
        shifted = (quotient << (i % 37)) + (remainder >> (i % 31))
        checksum = (
            checksum
            + shifted
            + modular
            + x.bit_length()
            + y.bit_length()
        ) % CHECK_MODULUS

        # Nursery digits must remain reachable from both live local payloads
        # and freshly boxed arithmetic results across repeated collections.
        if collect_every > 0 and i % collect_every == 0:
            gc.collect()

    return checksum


rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 500
collect_every = int(sys.argv[2]) if len(sys.argv) > 2 else 7
print(run(rounds, collect_every))
