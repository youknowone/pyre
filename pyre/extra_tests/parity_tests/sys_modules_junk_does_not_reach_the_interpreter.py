# CPython-suite gap: the suite pins the `None` sentinel as an import failure
# (`test_importlib.import_.test_api` `test_blocked_fromlist`) and it exercises
# `sys.stdout` redirection and `sys.unraisablehook` at length
# (`test_sys.test_unraisablehook`, `test_io`), but it never parks a non-module
# under `sys.modules['sys']` or `['builtins']` and then asks the interpreter to
# use its own `sys` -- and `test_sys.test_original_displayhook` and friends all
# run with the mapping intact.
#
# parity-tests reason: `sys.modules` is an ordinary dict the program owns, and
# a value parked under `sys` or `builtins` is not a statement about the
# interpreter's own module.  `_PySys_GetAttr` reads `PyInterpreterState.sysdict`
# and PyPy reads `space.sys`; neither goes through the mapping.  A runtime that
# does resolve its streams and hooks that way answers the program's junk from
# inside `print`, and one that also casts that value to a module writes through
# whatever word sits where a module's dict would be.
import gc
import io
import sys

real = sys.modules["sys"]

# The program's to make, and nothing downstream of it is the interpreter's.
sys.modules["sys"] = 42
sys.modules["builtins"] = 42

# `print` takes its target from the real module's `stdout` attribute, which is
# still the thing the program rebinds.
buf = io.StringIO()
sys.stdout = buf
print("captured")
sys.stdout = real.__stdout__
assert buf.getvalue() == "captured\n", buf.getvalue()


class Raiser:
    def __del__(self):
        raise RuntimeError("from __del__")


seen = []
sys.unraisablehook = lambda unraisable: seen.append(unraisable.exc_type.__name__)
Raiser()
gc.collect()
assert seen == ["RuntimeError"], seen

print("OK")

# Left parked through shutdown on purpose: finalization clears the
# interpreter's own `sys` and `builtins`, not what the mapping now holds.
