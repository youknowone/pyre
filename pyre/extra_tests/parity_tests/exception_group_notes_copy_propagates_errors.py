# CPython-suite gap: the except* corpus copies `__notes__` through `split()`
# only from plain lists, so nothing there reaches a `__notes__` whose lookup or
# whose conversion raises.
# parity-tests reason: two error paths the attribute copy used to swallow, where
# CPython and PyPy agree and only pyre differed.

"""A raising `__notes__` stops `subgroup`/`split` instead of being ignored.

`_derive_and_copy_attrs` reads `__notes__` off the original group with
`PyObject_GetOptionalAttr`, which reports only a *missing* attribute as absent
and propagates every other error, and converts a sequence with
`PySequence_List`, which propagates too.  Swallowing either one turns a user
error into a group that silently lost its notes.

Only a non-sequence is ignored, and that is deliberate: `__notes__` is supposed
to be a list, and `split()` is not a good place to report an error made earlier.
"""


def expect(exc_type, message, fn):
    try:
        fn()
    except exc_type as e:
        assert str(e) == message, (str(e), message)
        return
    raise AssertionError(f"{exc_type.__name__}({message!r}) was not raised")


def partial(notes):
    group = ExceptionGroup("m", [ValueError(1), TypeError(2)])
    group.__notes__ = notes
    return group


# A lookup that raises anything but AttributeError propagates.
class RaisingNotes(ExceptionGroup):
    @property
    def __notes__(self):
        raise ValueError("notes lookup boom")


raising = RaisingNotes("m", [ValueError(1), TypeError(2)])
expect(ValueError, "notes lookup boom", lambda: raising.subgroup(ValueError))
expect(ValueError, "notes lookup boom", lambda: raising.split(ValueError))

# An AttributeError from the same descriptor still reads as absent.
class AbsentNotes(ExceptionGroup):
    @property
    def __notes__(self):
        raise AttributeError("gone")


absent = AbsentNotes("m", [ValueError(1), TypeError(2)])
assert not hasattr(absent.subgroup(ValueError), "__notes__")


# A sequence whose conversion raises propagates.
class RaisingSequence:
    def __len__(self):
        return 2

    def __getitem__(self, index):
        raise ValueError("notes item boom")


expect(ValueError, "notes item boom", lambda: partial(RaisingSequence()).subgroup(ValueError))
expect(ValueError, "notes item boom", lambda: partial(RaisingSequence()).split(ValueError))

# A non-sequence is ignored, silently, and the copy still happens for sequences.
assert not hasattr(partial(5).subgroup(ValueError), "__notes__")
assert partial(("t",)).subgroup(ValueError).__notes__ == ["t"]
assert partial("ab").subgroup(ValueError).__notes__ == ["a", "b"]


# A `__getitem__`-only sequence is a sequence, and the copy is independent.
class IndexOnly:
    def __len__(self):
        return 2

    def __getitem__(self, index):
        if index >= 2:
            raise IndexError(index)
        return "s%d" % index


assert partial(IndexOnly()).subgroup(ValueError).__notes__ == ["s0", "s1"]

original = partial(["n1"])
taken = original.subgroup(ValueError)
assert taken.__notes__ == ["n1"]
assert taken.__notes__ is not original.__notes__
taken.__notes__.append("n2")
assert original.__notes__ == ["n1"]

print("OK")
