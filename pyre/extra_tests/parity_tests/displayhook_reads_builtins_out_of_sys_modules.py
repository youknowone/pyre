# pyre-check: pypy-diverges: pypy3's `displayhook` imports `builtins` rather
# than reading `sys.modules`, so a deleted key is re-created and a blocked one
# raises `ModuleNotFoundError`; neither refusal this pins is expressible there.
#
# CPython-suite gap: `test_sys.test_displayhook` covers the hook a program
# installs and the value `_` ends up holding; nothing in the suite takes
# `builtins` away from it.
#
# parity-tests reason: `sys_displayhook_impl` opens at `PyImport_GetModule`,
# ahead of the `None` test and ahead of any rendering, and binds `_` twice
# through whatever that lookup returned.  So a lost `builtins` is a refusal
# even for a value that is never rendered, and an entry that is not a module is
# refused by the binding rather than treated as absent.  A runtime that binds
# `_` best-effort writes the value out and then reports success for a hook that
# did not do what it says.
import sys


def refusal(fn):
    try:
        fn()
    except BaseException as exc:
        return type(exc).__name__, str(exc)
    return None


def with_entry(entry, value):
    saved = sys.modules["builtins"]
    sys.modules["builtins"] = entry
    try:
        return refusal(lambda: sys.displayhook(value))
    finally:
        sys.modules["builtins"] = saved


def without_entry(value):
    saved = sys.modules.pop("builtins")
    try:
        return refusal(lambda: sys.displayhook(value))
    finally:
        sys.modules["builtins"] = saved


# The lookup is the first step, so it refuses before the `None` test that would
# otherwise end the call.
LOST = ("RuntimeError", "lost builtins module")
assert without_entry(3) == LOST, without_entry(3)
assert without_entry(None) == LOST, without_entry(None)

# An entry that is present is handed to `setattr`, which states its own
# refusal -- and only once there is a value to bind, so `None` still returns.
assert with_entry(None, None) is None, with_entry(None, None)
assert with_entry(42, None) is None, with_entry(42, None)
for entry in (None, 42):
    got = with_entry(entry, 3)
    assert got is not None and got[0] == "AttributeError", (entry, got)

# Nothing above disturbed the ordinary path.
import builtins

sys.displayhook(7)
assert builtins._ == 7, builtins._
sys.displayhook(None)
assert builtins._ == 7, builtins._

print("OK")
