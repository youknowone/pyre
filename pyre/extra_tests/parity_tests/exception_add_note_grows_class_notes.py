# CPython-suite gap: the except* corpus and `test_exceptions` only ever
# `add_note` onto an instance that has no `__notes__` yet, so a class-level
# list being the object that grows is invisible there.
#
# parity-tests reason: `BaseException_add_note_impl` looks `__notes__` up
# with `PyObject_GetOptionalAttr` and `PyList_Append`s the list it found.
# A class attribute is that list, so the class and the instance share it
# afterwards. `W_BaseException.descr_add_note` goes through `getdict` and
# allocates an instance dict, so pypy3 leaves the class list alone.
#
# pyre-check: pypy-diverges: `descr_add_note` writes a new instance list
# via `getdict`, so pypy3 answers `inst.__notes__ == ['n']` and
# `E.__notes__ == ['class']` as two objects.
class E(ValueError):
    __notes__ = ["class"]


inst = E()
inst.add_note("n")
assert inst.__notes__ is E.__notes__, (inst.__notes__, E.__notes__)
assert inst.__notes__ == ["class", "n"], inst.__notes__

print("OK")
