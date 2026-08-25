# `object_getstate` calls an overriding `__getstate__` and only falls back to
# `object_getstate_default(required)` when the type still uses
# `object.__getstate__`.  The `required` refusal therefore stops at the
# boundary that hook draws: a native layout publishing one is rebuilt from the
# state it returns, and one that publishes none cannot be rebuilt through an
# empty `__newobj__` call.
#
# `__reduce_ex__` is called directly rather than through `pickle`: this
# directory carries its own `_pickle.py`, which shadows the stdlib extension
# module and stops `import pickle` from working here.  That keeps this file
# green under CPython too, unlike its `pickle_*` neighbours; it still carries
# no `gate=1` marker until it has been run against a build.
import io
import itertools
import types


def refuses(obj):
    try:
        obj.__reduce_ex__(2)
    except TypeError as e:
        assert "cannot pickle" in str(e), (obj, str(e))
        return True
    return False


# --- publishes `__getstate__`: reduces through the hook ---------------------
b = io.BytesIO(b"abcdef")
b.seek(2)
b.tag = "kept"
assert io.BytesIO.__getstate__ is not object.__getstate__
newobj, args, state, listitems, dictitems = b.__reduce_ex__(2)
assert args == (io.BytesIO,), args
assert state == (b"abcdef", 2, {"tag": "kept"}), state
assert listitems is None and dictitems is None, (listitems, dictitems)

s = io.StringIO("hello")
s.read(2)
assert io.StringIO.__getstate__ is not object.__getstate__
state = s.__reduce_ex__(2)[2]
# `(value, readnl, pos, dict)`; the trailing dict is empty here and pyre and
# CPython spell an empty one differently, so only value and pos are pinned.
assert state[0] == "hello", state
assert state[2] == 2, state

# --- publishes a refusing `__getstate__`: the hook owns the refusal ---------
w = io.BufferedWriter(io.BytesIO())
assert io.BufferedWriter.__getstate__ is not object.__getstate__
assert refuses(w)

# --- publishes none: `object_getstate_default(required)` refuses ------------
for obj in (
    types.ModuleType("m"),
    property(),
    staticmethod(len),
    classmethod(len),
    itertools.count(),
):
    assert type(obj).__getstate__ is object.__getstate__, obj
    assert refuses(obj), obj


# The refusal reaches neither an ordinary instance nor a list or a dict.
class C:
    def __init__(self):
        self.x = 1


assert C().__reduce_ex__(2)[2] == {"x": 1}
assert list(([1, 2]).__reduce_ex__(2)[3]) == [1, 2]
assert list(({"a": 1}).__reduce_ex__(2)[4]) == [("a", 1)]

print("pickle_native_getstate OK")
