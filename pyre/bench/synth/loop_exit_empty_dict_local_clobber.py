# pyre-check: max-pypy-ratio=5
# A local must not read back as the `for` loop's iterator after the loop.
#
# An empty dict literal used to decline in the codewriter, and the decline
# emitted `abort_permanent`, which closes its block: everything after it -
# the whole loop - stopped being lowered. The loop header still resolved to a
# walk entry, because the resume-marker tables carry an unlowered PC forward
# onto the last marker that WAS emitted, so the walk entered the prologue
# instead. Entry seeding fills a region's registers from the live frame by
# SLOT, and the frame is parked at the loop header, so the prologue's
# registers received the header's operand stack - the iterator - and replaying
# the prologue's STORE_FAST published it into a local.
#
# `seen` printed `range_iterator` instead of `set` once the loop crossed the
# compile threshold. The pre-compile iterations were correct and the local
# still read correctly from inside the loop body, so only the loop-exit path
# showed it.
#
# Discriminators, each verified against pypy3:
#   * only the EMPTY dict literal tripped it - `dict()`, `{1: 2}`, `[]`,
#     `set()`, `()` in the same position all lower normally;
#   * a `while` loop was clean, saved only by the walk-entry stack-depth check
#     that a `for` header's live stack of 1 happens to satisfy;
#   * assigning the dict inside the loop or under an `if` was clean, and so
#     was module scope - the function's own frame is what went wrong;
#   * any iterator reproduced it (`list_iterator` and `enumerate` landed in the
#     slot just as `range_iterator` did).
#
# Expected output: ('set', 'dict').
N = 4000


def driver(n):
    seen = set()
    acc = {}
    for i in range(n):
        pass
    return type(seen).__name__, type(acc).__name__


print(driver(N))
