# Pickler.persistent_id / Unpickler.persistent_load: the external-object hook.
# Both are settable on the instance and overridable on a subclass. A non-None
# persistent_id result is saved as a persistent reference (PERSID at protocol 0,
# BINPERSID at protocol >= 1) instead of by value; persistent_load resolves it
# back on load. Encountering a persistent reference without a persistent_load
# raises UnpicklingError. Pinned to CPython 3.14.
import io
import pickle


class Ref:
    def __init__(self, name):
        self.name = name


# A canonical object table keyed by name; the persistent id is the name.
registry = {"alpha": Ref("alpha"), "beta": Ref("beta")}
alpha = registry["alpha"]
beta = registry["beta"]


def pid(obj):
    if isinstance(obj, Ref):
        return obj.name
    return None


def pload(key):
    return registry[key]


# --- instance-attr hooks (the classic pattern) at both protocol families -----
for proto in (0, 2, 5):
    buf = io.BytesIO()
    p = pickle.Pickler(buf, proto)
    p.persistent_id = pid
    assert p.persistent_id is pid
    p.dump([alpha, beta, alpha, 42])

    u = pickle.Unpickler(io.BytesIO(buf.getvalue()))
    u.persistent_load = pload
    got = u.load()
    # Referenced objects resolve to the canonical instances; the int is by value.
    assert got[0] is alpha, proto
    assert got[1] is beta, proto
    assert got[2] is alpha, proto
    assert got[3] == 42, proto


# --- subclass overrides ------------------------------------------------------
class RefPickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, Ref):
            return obj.name
        return None


class RefUnpickler(pickle.Unpickler):
    def persistent_load(self, key):
        return registry[key]


buf = io.BytesIO()
RefPickler(buf, 2).dump([beta, 7])
got = RefUnpickler(io.BytesIO(buf.getvalue())).load()
assert got[0] is beta
assert got[1] == 7


# --- a persistent reference with no persistent_load is rejected ---------------
buf = io.BytesIO()
p = pickle.Pickler(buf, 2)
p.persistent_id = pid
p.dump([alpha])
data = buf.getvalue()
try:
    pickle.Unpickler(io.BytesIO(data)).load()
    raise AssertionError("expected UnpicklingError without persistent_load")
except pickle.UnpicklingError as e:
    assert "persistent_load" in str(e), e


# --- delete resets the instance hook -----------------------------------------
p = pickle.Pickler(io.BytesIO(), 2)
p.persistent_id = pid
del p.persistent_id
# After delete the hook is inactive again: nothing is saved by reference.
buf = io.BytesIO()
p2 = pickle.Pickler(buf, 2)
p2.persistent_id = pid
del p2.persistent_id
p2.dump([alpha])
# alpha is now pickled by value (no persistent_load needed to read it back).
back = pickle.loads(buf.getvalue())
assert isinstance(back[0], Ref) and back[0].name == "alpha"


print("_pickle_persistent OK")
