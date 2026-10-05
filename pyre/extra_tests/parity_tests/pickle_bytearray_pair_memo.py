# CPython-suite gap: `test.test_pickle.CPicklerTests.test_bytearray_memoization`
# dumps a repeated bytearray under every protocol; the suite does not pin a
# small nursery around BYTEARRAY8.
# parity-tests reason: `save_bytearray` must copy `view.as_str()` before
# BYTEARRAY8 / memoize, because the bytearray is nursery-movable.
# parity-env: PYPY_GC_NURSERY=8192
# parity-env: MAJIT_GC_NURSERY_POISON=1
import pickle

for proto in range(pickle.HIGHEST_PROTOCOL + 1):
    for payload in (b"", b"xyz", b"xyz" * 100):
        b = bytearray(payload)
        b1, b2 = pickle.loads(pickle.dumps((b, b), proto))
        assert b1 is b2
        assert bytes(b1) == payload
        a, c = bytearray(payload), bytearray(payload)
        d, e = pickle.loads(pickle.dumps((a, c), proto))
        assert d is not e
        assert bytes(d) == payload
        assert bytes(e) == payload

print("OK")
