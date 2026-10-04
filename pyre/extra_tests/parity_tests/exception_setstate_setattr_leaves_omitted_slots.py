# CPython-suite gap: `test_exceptions` pickles a fully-populated ImportError
# and never calls `__setstate__` with a partial dict, so an omitted path or
# a mutated state argument is invisible there.
#
# parity-tests reason: `BaseException___setstate___impl` setattr's each
# present key and leaves the others. `W_ImportError.descr_setstate` pops
# name/path/name_from with a None default, so an omitted path becomes None
# and the caller's state dict loses those keys. OSError has no own
# setstate; setattr of errno updates the member slot. PyPy's applevel
# setattr from `descr_setstate` writes errno into the instance dict.
#
# pyre-check: pypy-diverges: `W_ImportError.descr_setstate` pops omitted
# slots to None and mutates the state dict; `W_BaseException.descr_setstate`
# stores errno on the dict so `str(e)` keeps the constructor errno.
def import_error_omitted_path_stays():
    exc = ImportError("m", name="n", path="p")
    state = {"name": "z", "extra": 1}
    exc.__setstate__(state)
    assert exc.name == "z", exc.name
    assert exc.path == "p", exc.path
    assert exc.__dict__ == {"extra": 1}, exc.__dict__
    assert state == {"name": "z", "extra": 1}, state


def import_error_setstate_is_base():
    assert ImportError.__setstate__ is BaseException.__setstate__
    assert ModuleNotFoundError.__setstate__ is BaseException.__setstate__


def oserror_setstate_writes_errno_slot():
    exc = OSError(2, "m")
    exc.__setstate__({"errno": 9})
    assert exc.errno == 9, exc.errno
    assert str(exc) == "[Errno 9] m", str(exc)
    assert exc.args == (2, "m"), exc.args
    assert exc.__dict__ == {}, exc.__dict__


import_error_omitted_path_stays()
import_error_setstate_is_base()
oserror_setstate_writes_errno_slot()
print("OK")
