# CPython-suite gap: the except* corpus never forces a collection inside the
# projection walk, so nothing there reaches a relocated child.
# parity-tests reason: this guards a pyre/PyPy moving-GC allocation invariant.

"""`_exception_group_projection` keys on leaf ADDRESSES; nothing may move them.

`app_group.py _exception_group_projection` keeps the leaves to project in an
``identity_dict`` -- the objects themselves -- and then walks the original group
testing each child against that set.  pyre holds their addresses instead
(``ExceptionGroupCondition::Identity``), which is sound only while every leaf is
non-moving: ``w_exception_new_empty_impl`` allocates an exception through
``try_gc_alloc_stable_raw``, the oldgen, and ``check_new_args`` refuses a member
that is not a ``BaseException``.

This script is the guard on that invariant, not a reproducer -- it passes today
and is meant to.  It forces a collection from inside ``derive``, i.e. between
the leaf set being collected and the last child being tested, and asserts the
projection is still complete.  The day an exception is born in the nursery, the
address set starts dropping a relocated leaf and this fails.
"""

import gc


class Deriving(ExceptionGroup):
    """A group whose `derive` collects, the way an arbitrary override may."""

    def derive(self, excs):
        # The split rebuilds a subgroup through this hook, which is reached
        # after the identity set is collected and before the children that
        # follow this subgroup are tested.
        garbage = [[index] for index in range(2000)]
        assert len(garbage) == 2000
        gc.collect()
        return Deriving(self.message, excs)


def leaves(exc):
    found = []

    def walk(node):
        if isinstance(node, BaseExceptionGroup):
            for child in node.exceptions:
                walk(child)
        else:
            found.append(node)

    walk(exc)
    return found


for _ in range(20):
    # Empty the nursery first, so every leaf below is young -- and therefore
    # moved -- when `derive` collects mid-walk.
    gc.collect()
    inner_kept = ValueError("inner-kept")
    inner_dropped = TypeError("inner-dropped")
    outer_kept = ValueError("outer-kept")
    outer_dropped = TypeError("outer-dropped")
    group = Deriving(
        "outer",
        [Deriving("inner", [inner_kept, inner_dropped]), outer_kept, outer_dropped],
    )

    caught = None
    try:
        try:
            raise group
        except* ValueError:
            raise
        except* TypeError:
            pass
    except BaseExceptionGroup as reraised:
        caught = reraised

    assert caught is not None, "the reraised ValueError half must propagate"
    got = leaves(caught)
    # `inner_kept` is tested before any `derive` runs; `outer_kept` is tested
    # after the inner subgroup's two `derive` calls have each collected.
    assert any(leaf is inner_kept for leaf in got), got
    assert any(leaf is outer_kept for leaf in got), got
    assert not any(leaf is inner_dropped for leaf in got), got
    assert not any(leaf is outer_dropped for leaf in got), got

print("OK")
