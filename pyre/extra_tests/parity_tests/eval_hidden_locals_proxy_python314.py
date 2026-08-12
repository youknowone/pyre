"""CPython 3.14 eval locals while a PEP 709 hidden local is active."""

import sys


class Locals(dict):
    def __getitem__(self, key):
        if key == "sys":
            return sys
        return dict.__getitem__(self, key)

    def keys(self):
        raise AssertionError("eval locals must not back FrameLocalsProxy")


locals_mapping = Locals()
snapshots = eval("[locals() for i in (2, 3)]", {"sys": sys}, locals_mapping)
assert snapshots == [{"i": 2}, {"i": 3}], snapshots
assert locals_mapping == {}, locals_mapping

# A proxy write which is not a writable fast local belongs to the frame's
# separate f_extra_locals dict.  It is visible through the proxy alongside the
# live hidden local, without mutating the eval locals mapping.
observed = eval(
    "[(lambda p: (p.__setitem__('z', 3), dict(p)))"
    "(sys._getframe().f_locals) for i in (1,)]",
    {"sys": sys},
    locals_mapping,
)
assert observed == [(None, {"i": 1, "z": 3})], observed
assert locals_mapping == {}, locals_mapping

print("OK")
