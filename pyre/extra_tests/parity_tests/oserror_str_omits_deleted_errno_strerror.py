# CPython-suite gap: `test_exceptions` stringifies a constructed `OSError`
# and never deletes `errno` or `strerror`.
#
# parity-tests reason: `OSError_str` tests the `myerrno` and `strerror`
# pointers. `PyMember_SetOne` clears a `_Py_T_OBJECT` member, so after
# `del` those tests fail and `BaseException_str` prints the args tuple. An
# explicit `None` store stays present. Replacing `args` does not restamp
# the slots.
#
# pyre-check: pypy-diverges: `readwrite_attrproperty_w` installs no `fdel`,
# so `del e.strerror` raises AttributeError. `W_OSError.descr_str` is true
# for `space.w_None`.
def deleted_strerror_falls_back_to_args():
    exc = OSError(2, "m")
    del exc.strerror
    assert exc.strerror is None
    assert str(exc) == "(2, 'm')", str(exc)


def deleted_errno_falls_back_to_args():
    exc = OSError(2, "m")
    del exc.errno
    assert exc.errno is None
    assert str(exc) == "(2, 'm')", str(exc)


def deleted_both_fall_back_to_args():
    exc = OSError(2, "m")
    del exc.errno
    del exc.strerror
    assert str(exc) == "(2, 'm')", str(exc)


def explicit_none_strerror_stays_present():
    exc = OSError(2, "m")
    exc.strerror = None
    assert str(exc) == "[Errno 2] None", str(exc)


def explicit_none_errno_stays_present():
    exc = OSError(2, "m")
    exc.errno = None
    assert str(exc) == "[Errno None] m", str(exc)


def deleted_strerror_with_filename_renders_none():
    exc = OSError(2, "m", "a")
    del exc.strerror
    assert str(exc) == "[Errno 2] None: 'a'", str(exc)


def replacing_args_does_not_restamp_slots():
    exc = OSError(2, "m")
    exc.args = (9, "x")
    assert str(exc) == "[Errno 2] m", str(exc)


deleted_strerror_falls_back_to_args()
deleted_errno_falls_back_to_args()
deleted_both_fall_back_to_args()
explicit_none_strerror_stays_present()
explicit_none_errno_stays_present()
deleted_strerror_with_filename_renders_none()
replacing_args_does_not_restamp_slots()
print("OK")
