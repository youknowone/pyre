# pyre-check: gate=1
"""A width argument's `__index__` may resize a bytearray receiver.

`ljust`/`rjust`/`center`/`zfill`/`replace` coerce their arguments before
reading the receiver's bytes, so the result describes the receiver as it is
once every coercion has run.
"""


class Resize:
    def __init__(self, target, width):
        self.target = target
        self.width = width

    def __index__(self):
        self.target[:] = b"Z" * 64
        return self.width


for name in ("ljust", "rjust", "center", "zfill"):
    ba = bytearray(b"abcdefgh")
    got = getattr(ba, name)(Resize(ba, 40))
    assert got == b"Z" * 64, (name, bytes(got[:32]))

ba = bytearray(b"abcdefgh")
got = ba.replace(b"Z", b"y", Resize(ba, 3))
assert got == b"y" * 3 + b"Z" * 61, bytes(got[:32])
