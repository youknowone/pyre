import pickle
import pickletools
import sys


# Warm the nested optimize/framer paths into compiled code, then exercise the
# multi-frame carrier-abort shape that used to discard DoneWithThisFrameRef and
# make optimize() fall through with None.
for proto in range(pickle.HIGHEST_PROTOCOL + 1):
    for n in range(70):
        optimized = pickletools.optimize(pickle.dumps(2**n, proto))
        assert optimized
        len(optimized)

for proto in range(pickle.HIGHEST_PROTOCOL + 1):
    n = sys.maxsize
    while n:
        for expected in (-n, n):
            optimized = pickletools.optimize(pickle.dumps(expected, proto))
            assert pickle.loads(optimized) == expected
        n >>= 1

print("OK")
