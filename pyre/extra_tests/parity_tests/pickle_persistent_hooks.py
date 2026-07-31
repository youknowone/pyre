"""_pickle persistent hooks keep base methods and instance overrides separate."""

import _pickle
import io


pickler = _pickle.Pickler(io.BytesIO())
base_persistent_id = pickler.persistent_id
assert base_persistent_id("value") is None

seen = []


def persistent_id(obj):
    seen.append(obj)
    return None


pickler.persistent_id = persistent_id
assert pickler.persistent_id is persistent_id
pickler.dump("value")
assert seen == ["value"]
del pickler.persistent_id
assert pickler.persistent_id == base_persistent_id


class Pickler(_pickle.Pickler):
    def persistent_id(self, obj):
        assert super().persistent_id(obj) is None
        return None


Pickler(io.BytesIO()).dump("value")

unpickler = _pickle.Unpickler(io.BytesIO())
base_persistent_load = unpickler.persistent_load
try:
    base_persistent_load("pid")
except _pickle.UnpicklingError as exc:
    assert str(exc) == (
        "A load persistent id instruction was encountered, "
        "but no persistent_load function was specified."
    )
else:
    raise AssertionError("base persistent_load unexpectedly accepted a pid")


def persistent_load(pid):
    return pid


unpickler.persistent_load = persistent_load
assert unpickler.persistent_load is persistent_load
del unpickler.persistent_load
assert unpickler.persistent_load == base_persistent_load

print("OK")
