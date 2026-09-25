# pyre-check: gate=1
import atexit


def cb():
    pass


assert not hasattr(atexit.register, "__get__")
assert not hasattr(atexit.unregister, "__get__")
assert atexit.register.__module__ == "atexit"


class C:
    r = atexit.register


try:
    assert C().r(cb) is cb
finally:
    atexit.unregister(cb)
