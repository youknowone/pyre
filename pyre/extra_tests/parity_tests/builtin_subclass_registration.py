"""A builtin type registers itself on every entry of ``__bases__``.

``typeobject.py:1789-1790 TypeCache.ready`` runs ``w_type.ready()`` for each
builtin typedef the space cache builds, exactly as ``_type_new``
(``typeobject.py:970``) does for a heap type; ``ready`` walks the whole
``__bases__`` tuple and calls ``add_subclass`` on each entry
(``typeobject.py:1140-1142``).  So the number of bases is irrelevant: a
builtin appears in ``base.__subclasses__()`` for *every* base, not just the
primary one that drives its layout.

The two multi-base native types are ``ExceptionGroup(BaseExceptionGroup,
Exception)`` and ``io.UnsupportedOperation(OSError, ValueError)``.  They are
the only cases that distinguish "readied on the primary base" from "readied
on all bases", which is why they are pinned by name here.

``add_subclass`` (``typeobject.py:651-660``) is idempotent — it returns early
when an existing weakref already resolves to the subclass — so a type never
appears twice however many times it is readied.
"""

import io

MULTI_BASE = [
    (ExceptionGroup, ("BaseExceptionGroup", "Exception")),
    (io.UnsupportedOperation, ("OSError", "ValueError")),
]

for cls, base_names in MULTI_BASE:
    assert tuple(b.__name__ for b in cls.__bases__) == base_names, (
        cls.__qualname__,
        [b.__name__ for b in cls.__bases__],
    )
    for base in cls.__bases__:
        subs = base.__subclasses__()
        assert cls in subs, f"{cls.__qualname__} missing from {base.__name__}.__subclasses__()"
        assert subs.count(cls) == 1, (
            f"{cls.__qualname__} listed {subs.count(cls)}x in {base.__name__}.__subclasses__()"
        )

# A single-base builtin is registered by the same call, so the two paths must
# not disagree.
assert bool in int.__subclasses__()
assert int not in bool.__subclasses__()

# The multi-base entries are reachable by walking down from the base too, and
# the MRO orders the bases as recorded.
assert io.UnsupportedOperation in ValueError.__subclasses__()
assert io.UnsupportedOperation.__mro__[1:3] == (OSError, ValueError)
assert ExceptionGroup.__mro__[1:3] == (BaseExceptionGroup, Exception)

# `__subclasses__()` holds weak references, so a heap subclass that is still
# alive is listed and one that has been collected is not.
import gc


class _KeptOSError(OSError):
    pass


assert _KeptOSError in OSError.__subclasses__()


class _DroppedOSError(OSError):
    pass


del _DroppedOSError
gc.collect()
assert not any(b.__name__ == "_DroppedOSError" for b in OSError.__subclasses__())

# Readying does not disturb instantiation or the exception hierarchy.
err = io.UnsupportedOperation("nope")
assert isinstance(err, OSError)
assert isinstance(err, ValueError)
try:
    raise io.UnsupportedOperation("seek")
except ValueError as exc:
    assert type(exc) is io.UnsupportedOperation
else:
    raise AssertionError("UnsupportedOperation must be catchable as ValueError")

group = ExceptionGroup("g", [ValueError("v")])
assert isinstance(group, BaseExceptionGroup)
assert isinstance(group, Exception)

print("OK")
