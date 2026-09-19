# CPython-suite gap: the except* corpus never forces a collection inside the
# projection walk, so nothing there reaches a relocated child.
# parity-tests reason: this guards a pyre/PyPy moving-GC allocation invariant.

"""`_exception_group_projection` holds live leaf objects, not raw addresses.

`app_group.py _exception_group_projection` keeps the leaves to project in an
``identity_dict`` -- the objects themselves -- and then walks the original group
testing each child against that set.  pyre's ``ExceptionGroupCondition::Identity``
is that same set as a rooted list of leaves, so a collection inside ``derive``
forwards every not-yet-tested child instead of dropping it.

This script forces a collection from inside ``derive``, i.e. between the leaf
set being collected and the last child being tested, and asserts the projection
is still complete.
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
