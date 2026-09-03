# pyre-check: gate=1
"""Namespace paths re-read their mappings after the Python they dispatch.

`__build_class__`, `exec`, `__import__` and the mapping-pattern match each
resolve a namespace object up front and consume it several dispatches later.
Every one of those objects is nursery-allocated -- a dict, a tuple, or a
freshly built locals snapshot with no other referrer -- so the address held
across the dispatch is the pre-move one, and the snapshot is sweepable
besides.
"""

import gc

KEEP = None


def churn():
    """Take the freed boxes, so a sweep that frees is not invisible."""
    global KEEP
    KEEP = [[i] * 24 for i in range(60)] + [bytearray(b"Q" * 96) for _ in range(30)]


def collect():
    gc.collect()
    churn()


GLOBAL_MARK = "global-mark"


class Prepare(type):
    """`__prepare__` runs Python between the body function's decomposition and
    the frame that executes the class body."""

    @classmethod
    def __prepare__(mcls, name, bases, **kwds):
        collect()
        return dict(prepared=True)

    def __new__(mcls, name, bases, ns, **kwds):
        collect()
        return super().__new__(mcls, name, bases, ns)


def build_with_closure():
    cell_mark = "cell-mark"

    class Built(metaclass=Prepare):
        # Reads a global and a closure variable, so the frame must have been
        # given the function's live globals dict and closure tuple.
        seen_global = GLOBAL_MARK
        seen_cell = cell_mark
        prepared_seen = prepared  # noqa: F821 — from the prepared namespace

    return Built


Built = build_with_closure()
assert Built.seen_global == "global-mark", Built.seen_global
assert Built.seen_cell == "cell-mark", Built.seen_cell
assert Built.prepared_seen is True, Built.prepared_seen


class Entries:
    """`__mro_entries__` runs on every non-type base, and is handed the same
    original-bases tuple each round."""

    def __init__(self, *bases):
        self.bases = bases

    def __mro_entries__(self, orig_bases):
        collect()
        assert isinstance(orig_bases, tuple), orig_bases
        assert self in orig_bases, orig_bases
        return self.bases


class Left:
    pass


class Right:
    pass


class WithEntries(Entries(Left), Entries(Right)):
    pass


assert WithEntries.__bases__ == (Left, Right), WithEntries.__bases__
assert WithEntries.__orig_bases__[0].bases == (Left,)


class CollectingGlobals(dict):
    """A globals mapping whose `setdefault` runs Python, which is what
    `exec`/`eval` dispatch through before the frame exists."""

    def setdefault(self, key, default=None):
        collect()
        return super().setdefault(key, default)

    def __setitem__(self, key, value):
        collect()
        return super().__setitem__(key, value)


g = CollectingGlobals(marker="exec-globals")
exec("from_exec = marker", g)
assert g["from_exec"] == "exec-globals", g["from_exec"]

separate_locals = CollectingGlobals()
exec("in_locals = 41 + 1", g, separate_locals)
assert separate_locals["in_locals"] == 42, separate_locals


def exec_into_caller_locals():
    # `exec(src)` with neither globals nor locals resolves the caller's locals
    # into a fresh snapshot dict that nothing else holds.  The source reads two
    # of the caller's locals back out of it; a store into the snapshot itself
    # is invisible by design, so the result travels through `out`.
    local_mark = "local-mark"
    out = {}
    collect()
    exec("out['seen'] = local_mark + '/' + GLOBAL_MARK")
    return out["seen"]


assert exec_into_caller_locals() == "local-mark/global-mark"


class Level:
    def __index__(self):
        collect()
        return 0


imported = __import__("sys", globals(), locals(), ("path",), Level())
assert imported.path is not None


class TupleKey:
    KEY = (1, 2)


class CollectingMapping(dict):
    def get(self, key, default=None):
        collect()
        return super().get(key, default)


subject = CollectingMapping({TupleKey.KEY: "tuple-keyed", "plain": "str-keyed"})
match subject:
    case {TupleKey.KEY: found, "plain": also}:
        assert found == "tuple-keyed", found
        assert also == "str-keyed", also
    case _:
        raise AssertionError("mapping pattern did not match")
