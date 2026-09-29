# pyre-check: gate=1
"""`gc.enable` and `gc.collect` go through `enable_finalizers`, which raises.

`gc.disable` takes one finalizer lock.  An app-level `gc.enable_finalizers`
can release it first, and then the `enable_finalizers` call inside
`gc.collect` (for the duration of the drain) and inside `gc.enable` finds the
lock depth at zero and raises.  `gc.enable` has already switched the app-level
flag on by then.
"""

import gc

# `enable_finalizers` is a PyPy `gc` extension; CPython has no finalizer lock.
enable_finalizers = getattr(gc, "enable_finalizers", None)
if enable_finalizers is not None:
    gc.disable()
    enable_finalizers()

    try:
        gc.collect()
    except ValueError as error:
        assert str(error) == "finalizers are already enabled", error
    else:
        raise AssertionError("gc.collect must report the released lock")

    try:
        gc.enable()
    except ValueError as error:
        assert str(error) == "finalizers are already enabled", error
    else:
        raise AssertionError("gc.enable must report the released lock")
    assert gc.isenabled()

    gc.disable_finalizers()
    enable_finalizers()
