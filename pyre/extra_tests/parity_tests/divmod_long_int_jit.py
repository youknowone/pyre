# CPython-suite gap: divmod tests do not trace mixed bigint/int results under GC.
# parity-tests reason: this is the gc-poison owner for pyre's rooted bigint pair result.

"""Keep both bigint divmod halves live across allocation and collection.

The performance/fold census lives in ``bench/synth/divmod_long_int_pair.py``.
This small correctness leg remains here because the parity runner can enable
``MAJIT_GC_NURSERY_POISON``; the synth runner cannot.
"""

import gc

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

BIG = (1 << 200) + 12345
ROUNDS = 3000

held = []
for i in range(ROUNDS):
    divisor = 97 if i & 1 else -97
    q, r = divmod(BIG + i, divisor)
    assert q * divisor + r == BIG + i
    assert type(q) is int and type(r) is int
    held.append((i, divisor, q, r))
    if len(held) > 16:
        held.pop(0)
    if i % 256 == 0:
        gc.collect()
    for index, old_divisor, old_q, old_r in held:
        assert old_q * old_divisor + old_r == BIG + index

# Leave the bigint-pair arm for small quotients and rejected divisors without
# reusing its recorded result shape.
tiny = (1 << 70) >> 70
assert divmod(tiny, -7) == (-1, -6)
q, r = divmod(BIG, -(1 << 63))
assert q * -(1 << 63) + r == BIG
assert divmod(BIG, True) == (BIG, 0)
try:
    divmod(BIG, False)
except ZeroDivisionError:
    pass
else:
    raise AssertionError("divmod by False did not raise")

print("OK")
