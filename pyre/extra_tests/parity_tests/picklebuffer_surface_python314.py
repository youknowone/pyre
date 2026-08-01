"""CPython 3.14 exposes PickleBuffer through the pickle accelerator type."""

import _pickle


PB = _pickle.PickleBuffer
assert PB.__module__ == "pickle"
assert {"__buffer__", "__release_buffer__", "raw", "release"} <= set(PB.__dict__)
try:
    class Sub(PB):
        pass
except TypeError:
    pass
else:
    raise AssertionError("PickleBuffer accepted a subclass")

print("OK")
