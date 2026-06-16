# The `module` type is a real, registered type object: `type(m)` is a
# class (not the bare name string), instances carry `object`-inherited
# introspection, `module(name)` builds a working module, and pickling a
# module is refused at every protocol (its native name/dict payload
# cannot be reconstructed via `__newobj__`).
import sys
import types
import pickle

m = sys.modules["sys"]
M = type(m)

# A real type, not a string.
assert M is types.ModuleType, M
assert type(M) is type, type(M)
assert M.__name__ == "module", M.__name__
assert isinstance(m, object)
assert m.__class__ is M

# `module` defines its own `__new__` (its tp_new is not object's).
assert M.__new__ is not object.__new__

# Inherited object introspection resolves.
assert hasattr(m, "__reduce_ex__")
assert hasattr(m, "__dict__")
assert not (M.__flags__ & (1 << 7))  # module is instantiable, not DISALLOW

# Attribute access on a live module still works.
assert m.path is sys.path

# `module(name, doc=None)` builds a real, usable module.
fresh = M("fresh", "the docstring")
assert type(fresh) is M
assert fresh.__name__ == "fresh"
assert fresh.__doc__ == "the docstring"
assert fresh.__spec__ is None
assert repr(fresh) == "<module 'fresh'>", repr(fresh)
fresh.value = 42
assert fresh.value == 42
assert vars(fresh)["value"] == 42

# A module cannot be pickled at any protocol.
for proto in range(0, pickle.HIGHEST_PROTOCOL + 1):
    try:
        pickle.dumps(m, proto)
    except TypeError as e:
        assert str(e) == "cannot pickle 'module' object", str(e)
    else:
        raise AssertionError(("module should not pickle", proto))

print("module_type OK")
