# CPython-suite gap: the except* corpus asks what a subgroup contains, never
# whether an unfiltered one is the group it was taken from, so nothing there
# reaches the arm this pins.
# parity-tests reason: an identity and an answered class where PyPy differs from
# CPython on purpose.
# pyre-check: pypy-diverges: pins that an unfiltered `subgroup` is rebuilt;
# `app_group.py subgroup` returns `self` there on purpose ("this is the
# difference to split!"), so pypy3 answers `eg.subgroup(cond) is eg` True and a
# `BaseExceptionGroup` subclass rather than `ExceptionGroup`.

"""`subgroup` rebuilds its result even when the walk dropped nothing.

`exceptiongroup_subgroup` returns the group itself only when the condition
matches that group; once it walks the children it goes through `derive`
whatever the walk found.  So the result of an all-matching `subgroup` is a new
object, and for a `BaseExceptionGroup` subclass the default `derive` builds a
`BaseExceptionGroup`, which is promoted to `ExceptionGroup` because every leaf
is an `Exception`.

Both facts are observable — `is` and `type()` — and a shortcut that answers the
receiver is invisible to any test that only reads `.exceptions`.
"""


class MyEG(BaseExceptionGroup):
    """A subclass that adds nothing, so only `derive` decides the result type."""


value = ValueError("v")
kind = TypeError("t")

# The one arm that does answer the receiver: the condition matches the group.
group = MyEG("m", [value])
assert group.subgroup(MyEG) is group
assert group.subgroup(BaseExceptionGroup) is group
assert group.subgroup(lambda exc: True) is group
assert type(group.subgroup(MyEG)) is MyEG

# Every leaf matches, so the walk drops nothing -- and the result is still new.
plain = ExceptionGroup("m", [value])
taken = plain.subgroup(ValueError)
assert taken is not plain
assert taken.message == "m"
assert taken.exceptions == (value,)

# The same for a subclass, where the rebuild is visible in the class as well.
assert type(group.subgroup(ValueError)) is ExceptionGroup
assert group.subgroup(ValueError).exceptions == (value,)

# Nested: the inner group is unfiltered too, and is rebuilt with the outer.
inner = ExceptionGroup("i", [value])
outer = ExceptionGroup("o", [inner])
rebuilt = outer.subgroup(ValueError)
assert rebuilt is not outer
assert rebuilt.exceptions[0] is not inner
assert rebuilt.exceptions[0].exceptions == (value,)

# A partial match was already rebuilt before, and stays so.
mixed = MyEG("m", [value, kind])
part = mixed.subgroup(ValueError)
assert part is not mixed
assert type(part) is ExceptionGroup
assert part.exceptions == (value,)

# `split` never took the shortcut on either side.
assert plain.split(ValueError)[0] is not plain
assert plain.split(KeyError)[1] is not plain

# The rebuild runs `_derive_and_copy_attrs`, so the metadata follows it.
noted = ExceptionGroup("m", [value])
noted.add_note("n1")
assert noted.subgroup(ValueError).__notes__ == ["n1"]
assert noted.subgroup(ValueError).__notes__ is not noted.__notes__

print("OK")
