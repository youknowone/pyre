# pyre-check: gate=1
"""OSError stores the arguments it was given, not the ones it started with.

Building the message runs every argument's `__repr__`, and the tuple is
stored afterwards, so an argument that moved while a `__repr__` ran must
still be stored correctly.
"""


class R:
    def __repr__(self):
        for _ in range(300):
            [0] * 64
        return "R"


for _ in range(200):
    payload = [1, 2, 3]
    err = OSError(payload, R())
    assert err.args[0] == [1, 2, 3], repr(err.args)
    assert err.args[0] is payload, repr(err.args)
    assert repr(err.args[1]) == "R", repr(err.args)
