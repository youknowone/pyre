# CPython-suite gap: generator tests do not cover function-name object reuse.
# parity-tests reason: PyPy stores the function's existing names on a new generator.

"""Generator construction retains the function's current immutable names."""


def items():
    yield 1


name = "renamed"
qualname = "qualified.items"
items.__name__ = name
items.__qualname__ = qualname
generator = items()

assert generator.__name__ is name
assert generator.__qualname__ is qualname

items.__name__ = "later"
items.__qualname__ = "later.items"
assert generator.__name__ == "renamed"
assert generator.__qualname__ == "qualified.items"

surrogate = "items\ud800"
items.__name__ = surrogate
items.__qualname__ = surrogate
surrogate_generator = items()
assert surrogate_generator.__name__ is surrogate
assert surrogate_generator.__qualname__ is surrogate

print("OK")
