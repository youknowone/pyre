# CPython-suite gap: `test_exceptions` deletes `__cause__` / `__context__` /
# `__traceback__` and never `__suppress_context__`, and only writes True/False
# to it. It only ever `add_note`s onto an instance that has no `__notes__`
# yet, so a class-level list being the object that grows is invisible. It
# reads `StopIteration.value` on a freshly constructed instance and never
# calls `__init__` again with no arguments.
#
# parity-tests reason: `__suppress_context__` is a `T_BOOL` member, so
# `PyMember_SetOne` refuses a delete with `can't delete numeric/char
# attribute` and a non-bool store with `attribute value type must be bool`;
# `descr_delsuppresscontext` / `descr_setsuppresscontext` word both
# differently. `BaseException_add_note_impl` looks `__notes__` up with
# `PyObject_GetOptionalAttr` and `PyList_Append`s the list it found, so a
# class attribute is that list; `W_BaseException.descr_add_note` goes through
# `getdict` and allocates an instance list. `StopIteration_init` always
# `Py_CLEAR`s `value` and then stores the first positional argument or
# `None`; `W_StopIteration.descr_init` writes `w_value` only when `args_w` is
# non-empty.
#
# pyre-check: pypy-diverges: pypy3 reports `__suppress_context__ may not be
# deleted` and `expected integer, got str object`, leaves a class-level
# `__notes__` list alone (`inst.__notes__ == ['n']`), and keeps
# `e.value is 1` after `StopIteration(1).__init__()`.
def rejects(fn, message):
    try:
        fn()
    except TypeError as err:
        assert str(err) == message, err
    else:
        raise AssertionError("expected TypeError")


def del_suppress_context(e):
    del e.__suppress_context__


def del_cause(e):
    del e.__cause__


def store_suppress_context(e):
    e.__suppress_context__ = "yes"


e = ValueError()
rejects(lambda: del_suppress_context(e), "can't delete numeric/char attribute")
rejects(lambda: del_cause(e), "__cause__ may not be deleted")
rejects(lambda: store_suppress_context(e), "attribute value type must be bool")


class E(ValueError):
    __notes__ = ["class"]


inst = E()
inst.add_note("n")
assert inst.__notes__ is E.__notes__, (inst.__notes__, E.__notes__)
assert inst.__notes__ == ["class", "n"], inst.__notes__

e = StopIteration(1)
e.__init__()
assert e.value is None, e.value
assert e.args == (), e.args

e = StopIteration(1)
e.__init__(2)
assert e.value == 2, e.value
assert e.args == (2,), e.args

print("OK")
