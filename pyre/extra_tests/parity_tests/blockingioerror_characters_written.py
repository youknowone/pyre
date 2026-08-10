for count in range(6):
    error = BlockingIOError(*("a", "b", "c", "d", "e")[:count])
    try:
        error.characters_written
    except AttributeError:
        pass
    else:
        raise AssertionError("unset characters_written must be absent")

error = BlockingIOError("a", "b", 3)
assert error.args == ("a", "b", 3)
assert error.characters_written == 3
error.characters_written = 5
assert error.characters_written == 5
assert error.args == ("a", "b", 3)
del error.characters_written
try:
    error.characters_written
except AttributeError:
    pass
else:
    raise AssertionError("deleted characters_written must be absent")

plain = OSError()
plain.characters_written = 7
assert plain.characters_written == 7
del plain.characters_written

generic = Exception()
assert not hasattr(generic, "characters_written")
generic.characters_written = 11
assert generic.__dict__["characters_written"] == 11
del generic.characters_written
assert not hasattr(generic, "characters_written")

print("OK")
