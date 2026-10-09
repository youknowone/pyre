# CPython-suite gap: `test_unpack_ex` and `test_extcall` check what a starred
# display and a `f(x, *args)` call produce, once per shape. None of them runs
# the display often enough to be compiled, and the lost item below only shows
# on the iteration that traces it.
#
# parity-tests reason: `[x, *args]` builds a one-item list and extends it from
# the tuple, so the append has to grow the items block
# (`_ll_list_resize_ge` -> `_ll_list_resize_hint_really`) before
# `ll_setitem_fast` stores the item. A tracer that sets the new length but
# leaves the store out of the grown block hands back a list whose second slot
# is empty, and the call built from it reports a missing argument.
#
# CPython 3.14 and PyPy agree on every arm below.
ITEM = ["a", "b"]


def display_list(*args):
    return [len, *args]


def display_tuple(*args):
    return (len, *args)


def forward(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def target(args=None, namespace=None):
    return len(args)


def call_through(*args, **kwargs):
    return forward(target, *args, **kwargs)


def a_starred_list_display_keeps_the_unpacked_item():
    for i in range(3000):
        r = display_list(ITEM)
        assert len(r) == 2 and r[0] is len and r[1] is ITEM, (i, len(r))


def a_starred_tuple_display_keeps_the_unpacked_item():
    for i in range(3000):
        r = display_tuple(ITEM)
        assert len(r) == 2 and r[0] is len and r[1] is ITEM, (i, len(r))


def a_forwarded_star_call_keeps_its_positional_argument():
    total = 0
    for i in range(3000):
        total += call_through(ITEM)
    assert total == 6000, total


a_starred_list_display_keeps_the_unpacked_item()
a_starred_tuple_display_keeps_the_unpacked_item()
a_forwarded_star_call_keeps_its_positional_argument()
print("OK")
