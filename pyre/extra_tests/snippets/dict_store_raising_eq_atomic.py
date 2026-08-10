"""A dict store that raises during probing is atomic.

CPython's ``test_bad_key`` checks propagation but not that an earlier colliding
entry and old value survive. This generic dict contract belongs in snippets.
"""

g = {}
_armed = False


class _Raiser:
    def __hash__(self):
        return 7

    def __eq__(self, other):
        if _armed:
            raise ValueError("boom")
        return self is other


class _Quiet:
    def __hash__(self):
        return 7

    def __eq__(self, other):
        return self is other


def _store_must_not_mutate_on_raise():
    global _armed
    quiet = _Quiet()
    g[_Raiser()] = "raiser"
    g[quiet] = "quiet-old"
    before = list(g.items())
    _armed = True
    try:
        g[quiet] = "quiet-new"
    except ValueError:
        pass
    else:
        assert False, "raising __eq__ must propagate out of the store"
    finally:
        _armed = False
    live = {id(k) for k, _ in g.items()}
    dropped = [k for k, _ in before if id(k) not in live]
    assert not dropped, f"store dropped unrelated keys: {dropped!r}"
    assert len(g) == len(before), f"len {len(before)} -> {len(g)}"
    assert g[quiet] == "quiet-old", f"value applied despite raise: {g[quiet]!r}"


_store_must_not_mutate_on_raise()


print("OK")
