# CPython-suite gap: the except* corpus only ever gives a group a list of notes,
# so nothing there distinguishes "is a sequence" from "is iterable".
# parity-tests reason: the sequence test that decides whether `__notes__` is
# copied at all, where PyPy's app-level `list()` accepts more than CPython's
# `PySequence_Check` does.
# pyre-check: pypy-diverges: `app_group.py _derive_and_copy_attrs` copies
# `__notes__` with `list(self.__notes__)`, which accepts any iterable and runs
# its iterator, so pypy3 copies a generator's items and propagates a ValueError
# out of a raising `__iter__`; both are ignored here.

"""`__notes__` is copied only when it is a *sequence*, not merely iterable.

The gate is `PySequence_Check` — `issequence_w`, i.e. `__getitem__` on something
that is not a mapping — and a value that fails it is ignored without being
iterated at all.  A generator and an object carrying only `__iter__` therefore
leave the derived group with no notes, and a raising `__iter__` never runs.

`list()` on the same values would copy the generator and raise out of the
`__iter__`, which is what makes this observable rather than an internal detail.
"""


def partial(notes):
    group = ExceptionGroup("m", [ValueError(1), TypeError(2)])
    group.__notes__ = notes
    return group


# A generator is iterable but not a sequence: no notes on the result.
assert not hasattr(partial(iter(["g"])).subgroup(ValueError), "__notes__")
yes, no = partial(iter(["g"])).split(ValueError)
assert not hasattr(yes, "__notes__")
assert not hasattr(no, "__notes__")


# An object with only `__iter__` is not a sequence either, so the iterator is
# never run and its error never surfaces.
class RaisingIter:
    def __iter__(self):
        raise ValueError("iter boom")


assert not hasattr(partial(RaisingIter()).subgroup(ValueError), "__notes__")
yes, no = partial(RaisingIter()).split(ValueError)
assert not hasattr(yes, "__notes__")
assert not hasattr(no, "__notes__")

# A mapping defines `__getitem__` but is not a sequence: also ignored.
assert not hasattr(partial({"k": "v"}).subgroup(ValueError), "__notes__")

print("OK")
