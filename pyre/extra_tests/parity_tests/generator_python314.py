"""PyPy GeneratorIterator typedef with Python 3.14 concrete slots."""

import weakref


def inner():
    yield 10


def sample():
    yield 1
    yield from inner()


generator = sample()
generator_type = type(generator)
assert {
    "__class_getitem__",
    "__del__",
    "__doc__",
    "__iter__",
    "__name__",
    "__next__",
    "__qualname__",
    "__repr__",
    "__sizeof__",
    "close",
    "gi_code",
    "gi_frame",
    "gi_running",
    "gi_suspended",
    "gi_yieldfrom",
    "send",
    "throw",
} <= set(generator_type.__dict__)
assert generator_type.__doc__ is None
assert iter(generator) is generator
assert generator.__name__ == "sample"
assert generator.__qualname__.endswith("sample")
assert generator.gi_code.co_name == "sample"
assert generator.gi_frame is not None
assert generator.gi_running is False and generator.gi_suspended is False
assert generator.gi_yieldfrom is None
assert "generator object" in repr(generator)
assert generator.__sizeof__() > 0
assert weakref.ref(generator)() is generator

generator.__name__ = "renamed"
generator.__qualname__ = "qualified.renamed"
assert generator.__name__ == "renamed"
assert generator.__qualname__ == "qualified.renamed"
assert "qualified.renamed" in repr(generator)
for attribute in ("__name__", "__qualname__"):
    try:
        setattr(generator, attribute, 42)
    except TypeError:
        pass
    else:
        raise AssertionError("generator names must remain strings")

assert next(generator) == 1
assert generator.gi_suspended is True
assert next(generator) == 10
assert generator.gi_yieldfrom is not None
try:
    next(generator)
except StopIteration:
    pass
else:
    raise AssertionError("generator must finish")
assert generator.gi_frame is None
assert generator.gi_suspended is False

alias = generator_type[int, str, float]
assert alias.__origin__ is generator_type

closable = sample()
closable.__del__()
try:
    next(closable)
except StopIteration:
    pass
else:
    raise AssertionError("generator.__del__ must close the generator")

print("OK")
