# CPython-suite gap: `test_functools` covers what `partial` does and how it
# reprs, never what its own methods are called, so a helper standing in for a
# method reads as a pass.
#
# parity-tests reason: `partial.__new__` and `partial.__repr__` are the class's
# own methods, and a traceback, a `help()` page and every "missing argument"
# message name them through `__qualname__`.  An accelerator that assigns a
# module-level helper into the class instead reports the helper's private name
# for a call the program made through the class.
import functools

partial = functools.partial

assert partial.__new__.__name__ == "__new__", partial.__new__.__name__
assert partial.__new__.__qualname__ == "partial.__new__", partial.__new__.__qualname__
assert partial.__repr__.__name__ == "__repr__", partial.__repr__.__name__
assert partial.__repr__.__qualname__ == "partial.__repr__", partial.__repr__.__qualname__

# The names are the only thing under test; the behaviour behind them is not
# disturbed by where they are written.
doubled = partial(pow, 2)
assert doubled(3) == 8
assert repr(doubled).startswith("functools.partial("), repr(doubled)
assert partial(doubled, 4).func is pow


class Sub(partial):
    pass


assert Sub.__new__.__qualname__ == "partial.__new__", Sub.__new__.__qualname__
assert Sub(pow, 2)(3) == 8

print("OK")
