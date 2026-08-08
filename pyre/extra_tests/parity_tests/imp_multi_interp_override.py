"""`_imp._override_multi_interp_extensions_check` exists and refuses the main interpreter.

The name overrides `PyInterpreterConfig.check_multi_interp_extensions` for a
subinterpreter. Called from the main interpreter it raises `RuntimeError`, so
`importlib.util._incompatible_extension_module_restrictions` — which is written
entirely in terms of that call (`importlib/util.py:154`, `:160`) — cannot be
entered here either. Both are asserted, because the failure this replaced was an
`AttributeError` from the name being absent, which is a different diagnosis for
anything that catches it.
"""

import _imp
import importlib.util

MESSAGE = (
    "_imp._override_multi_interp_extensions_check() cannot be used in the main interpreter"
)

assert "_override_multi_interp_extensions_check" in dir(_imp)

try:
    _imp._override_multi_interp_extensions_check(1)
except RuntimeError as exc:
    assert str(exc) == MESSAGE, str(exc)
else:
    raise AssertionError("the main interpreter must be refused")

# Every override value is refused the same way, including the 0 "no override".
for value in (-1, 0, 1, 7, True):
    try:
        _imp._override_multi_interp_extensions_check(value)
    except RuntimeError as exc:
        assert str(exc) == MESSAGE, (value, str(exc))
    else:
        raise AssertionError(f"{value!r} must be refused")

# The argument is converted before the interpreter is checked, so an argument
# error still reports as one.
try:
    _imp._override_multi_interp_extensions_check("x")
except TypeError:
    pass
else:
    raise AssertionError("a str argument must raise TypeError")

try:
    _imp._override_multi_interp_extensions_check()
except TypeError:
    pass
else:
    raise AssertionError("a no-argument call must raise TypeError")

try:
    _imp._override_multi_interp_extensions_check(1, 2)
except TypeError:
    pass
else:
    raise AssertionError("a two-argument call must raise TypeError")

# The context manager reaches the same refusal on `__enter__`, for either
# `disable_check` value, and leaves nothing behind.
for disable in (True, False):
    manager = importlib.util._incompatible_extension_module_restrictions(
        disable_check=disable
    )
    assert manager.override == (-1 if disable else 1), manager.override
    try:
        with manager:
            raise AssertionError("the context manager must not be enterable")
    except RuntimeError as exc:
        assert str(exc) == MESSAGE, str(exc)
    assert not hasattr(manager, "old")

print("OK")
