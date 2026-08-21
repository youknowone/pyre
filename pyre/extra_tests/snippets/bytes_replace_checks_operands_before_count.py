# pyre-check: gate=1
"""`bytes.replace` rejects a bad `old`/`new` before it coerces `count`.

The clinic signature converts the two buffers ahead of the integer, so a
non-bytes-like operand raises `TypeError` without ever reaching the third
argument's `__index__`.  The payload slices are still taken after the
coercion, because a `bytearray` operand resized there reallocates the buffer
they point into.
"""


class Counter:
    def __init__(self):
        self.calls = 0

    def __index__(self):
        self.calls += 1
        return 1


for receiver in (b"abcabc", bytearray(b"abcabc")):
    for old, new in ((1, b"z"), (b"a", 2), (None, None)):
        count = Counter()
        try:
            receiver.replace(old, new, count)
        except TypeError as exc:
            print(type(receiver).__name__, type(exc).__name__, exc)
        else:
            print(type(receiver).__name__, "no error")
        print("  __index__ calls:", count.calls)


class Grow:
    def __init__(self, target):
        self.target = target

    def __index__(self):
        self.target += b"Q" * 64
        return 2


target = bytearray(b"abcabcabc")
print(target.replace(b"a", b"zz", Grow(target)))
print(len(target))
