import pickle


exc = Exception(42)
exc.blah = 53
exc.__setstate__({"args": (1, 2, 3), "blah": 35, "extra": 11})
assert exc.args == (1, 2, 3)
assert exc.blah == 35
assert exc.extra == 11


class WatchedError(Exception):
    def __setattr__(self, name, value):
        if name != "seen":
            object.__setattr__(self, "seen", name)
        object.__setattr__(self, name, value)


watched = WatchedError()
watched.__setstate__({"restored": 7})
assert watched.seen == "restored"
assert watched.restored == 7


attribute_error = AttributeError("missing", name="field", obj=lambda: None)
attribute_error.detail = "kept"
state = attribute_error.__getstate__()
assert state == {
    "detail": "kept",
    "name": "field",
    "args": ("missing",),
}
assert "obj" not in state

assert AttributeError().__getstate__() == {"args": ()}
assert AttributeError(name=None).__getstate__() == {"name": None, "args": ()}

for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
    restored = pickle.loads(pickle.dumps(attribute_error, protocol))
    assert restored.args == ("missing",)
    assert restored.name == "field"
    assert restored.obj is None
    assert restored.detail == "kept"

print("OK")
