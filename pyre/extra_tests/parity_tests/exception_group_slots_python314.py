"""`BaseExceptionGroup` keeps message/exceptions off the instance dictionary.

`interp_group.py:19-20 descr_new` stamps `w_message` / `w_exceptions` onto the
instance and `:71-72` exposes them as read-only attrproperties, so the public
`__dict__` is untouched and a dictionary poke cannot rewrite either one.
"""

group = ExceptionGroup("m", [ValueError("v")])
assert group.__dict__ == {}
assert vars(group) == {}
assert group.message == "m"
assert isinstance(group.exceptions, tuple)
assert len(group.exceptions) == 1
assert isinstance(group.exceptions[0], ValueError)

for attribute in ("message", "exceptions"):
    try:
        setattr(group, attribute, "x")
    except AttributeError:
        pass
    else:
        raise AssertionError(f"{attribute} must be read-only")

group.__dict__["message"] = "poked"
group.__dict__["exceptions"] = ()
assert group.message == "m"
assert len(group.exceptions) == 1

group.annotation = 1
assert group.__dict__ == {"message": "poked", "exceptions": (), "annotation": 1}

# `exceptions` is the tuple flattened at construction, so it does not follow a
# later mutation of the list the caller passed — which `args` keeps by
# reference.
supplied = [ValueError("a"), TypeError("b")]
mutated = ExceptionGroup("two", supplied)
supplied.append(OSError("c"))
assert len(mutated.exceptions) == 2
assert mutated.args[1] is supplied
assert len(mutated.args[1]) == 3
assert str(mutated) == "two (2 sub-exceptions)"

base = BaseExceptionGroup("b", (KeyboardInterrupt(),))
assert base.__dict__ == {}
assert base.message == "b"
assert len(base.exceptions) == 1

print("OK")
