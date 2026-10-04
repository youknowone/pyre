# CPython-suite gap: `test_exceptions` pickles a constructed ImportError
# or AttributeError and never deletes `name` afterwards.
#
# parity-tests reason: `ImportError_getstate` copies every non-NULL slot,
# explicit `None` included. `AttributeError_getstate` copies a non-NULL
# `name` the same way. `PyMember_SetOne` clears `_Py_T_OBJECT` to NULL, so
# `del e.name` drops the key from the reduce state while `e.name = None`
# keeps it.
#
# pyre-check: pypy-diverges: `readwrite_attrproperty_w` installs no `fdel`,
# so `del e.name` raises AttributeError. `W_ImportError.descr_reduce` skips
# `is_w(None)`.
def import_error_deleted_name_drops_from_state():
    exc = ImportError("m", name="n", path="p")
    del exc.name
    assert exc.name is None
    state = exc.__reduce__()[2]
    assert state == {"path": "p"}, state


def import_error_assigned_none_name_stays_in_state():
    exc = ImportError("m", name="n", path="p")
    exc.name = None
    state = exc.__reduce__()[2]
    assert state == {"name": None, "path": "p"}, state


def import_error_deleted_path_drops_from_state():
    exc = ImportError("m", name="n", path="p")
    del exc.path
    state = exc.__reduce__()[2]
    assert state == {"name": "n"}, state


def attribute_error_deleted_name_drops_from_state():
    exc = AttributeError("m", name="n")
    del exc.name
    assert exc.name is None
    state = exc.__reduce__()[2]
    assert "name" not in state, state
    assert state["args"] == ("m",)


def attribute_error_assigned_none_name_stays_in_state():
    exc = AttributeError("m", name="n")
    exc.name = None
    state = exc.__reduce__()[2]
    assert state["name"] is None, state


import_error_deleted_name_drops_from_state()
import_error_assigned_none_name_stays_in_state()
import_error_deleted_path_drops_from_state()
attribute_error_deleted_name_drops_from_state()
attribute_error_assigned_none_name_stays_in_state()
print("OK")
