# CPython-suite gap: `test_int` hashes small ints and does not check that two
# independently allocated large integers with the same value share a hash, or
# that dict/set lookup follows that hash.
#
# Equal integer values hash equal, including values that only exist as
# `W_LongObject` and values that fit i64 after bigint arithmetic.
#
# parity-tests reason: `try_hash_value` used to read a bigint as `W_IntObject`
# and hash the BigInt pointer; equal values then disagreed.

print(hash(2**100) == hash(2**100 + 0))
print(hash(2**100))
print(hash(-(2**70)))
print({2**100: "a"}[2**99 * 2])
print((2**100 + 0) in {2**100})
print(hash((2**100) // (2**60)) == hash((2**100) // (2**60) + 0))
print(hash((2**100) // (2**60)))
print({(2**100) // (2**60): "b"}[((2**100) // (2**60)) + 0])
print(((2**100) // (2**60) + 0) in {(2**100) // (2**60)})
print("OK")
