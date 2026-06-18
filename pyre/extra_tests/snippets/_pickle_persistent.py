# Pickler.persistent_id / Unpickler.persistent_load: the external-object hook.
# pickletester (test_persistence) already covers the persistent-reference
# round-trip across protocols. What we pin here are the instance-attribute
# mechanics CPython's suite drives only via subclasses:
#   - the hooks are settable on the instance and `del` resets them (after which
#     nothing is saved by reference again),
#   - a persistent reference loaded with no persistent_load raises
#     UnpicklingError with a message mentioning persistent_load.
# Pinned to CPython 3.14.
import io
import pickle


class Ref:
    def __init__(self, name):
        self.name = name


registry = {"alpha": Ref("alpha"), "beta": Ref("beta")}
alpha = registry["alpha"]


def pid(obj):
    if isinstance(obj, Ref):
        return obj.name
    return None


def pload(key):
    return registry[key]


# --- instance-attr hooks are settable and round-trip by reference ------------
buf = io.BytesIO()
p = pickle.Pickler(buf, 2)
p.persistent_id = pid
assert p.persistent_id is pid
p.dump([alpha, 42])
u = pickle.Unpickler(io.BytesIO(buf.getvalue()))
u.persistent_load = pload
got = u.load()
assert got[0] is alpha
assert got[1] == 42


# --- a persistent reference with no persistent_load is rejected ---------------
buf = io.BytesIO()
p = pickle.Pickler(buf, 2)
p.persistent_id = pid
p.dump([alpha])
try:
    pickle.Unpickler(io.BytesIO(buf.getvalue())).load()
    raise AssertionError("expected UnpicklingError without persistent_load")
except pickle.UnpicklingError as e:
    assert "persistent_load" in str(e), e


# --- delete resets the instance hook -----------------------------------------
buf = io.BytesIO()
p2 = pickle.Pickler(buf, 2)
p2.persistent_id = pid
del p2.persistent_id
# After delete the hook is inactive: alpha is pickled by value (no
# persistent_load needed to read it back).
p2.dump([alpha])
back = pickle.loads(buf.getvalue())
assert isinstance(back[0], Ref) and back[0].name == "alpha"


print("_pickle_persistent OK")
