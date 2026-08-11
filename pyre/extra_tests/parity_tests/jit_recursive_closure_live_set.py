"""A recursive closure must keep its freshly captured set live across a call."""

import _pylong


def expected_powers(w, base, limit, need_hi):
    seen = set()
    need = set()

    def visit(w):
        if w <= limit or w in seen:
            return
        seen.add(w)
        lo = w >> 1
        hi = w - lo
        need.add(hi if need_hi else lo)
        visit(lo)
        visit(hi)

    visit(w)
    # Keeping only the closure-owned reference live across this residual call
    # used to make the compiled outer frame reload an incomplete set.
    actual = _pylong.compute_powers(w, base, limit, need_hi=need_hi)
    assert actual.keys() == need, (w, actual.keys(), need)


for base in (2, 5):
    for need_hi in (False, True):
        for limit in (1, 11):
            for width in range(250, 550):
                expected_powers(width, base, limit, need_hi)

print("OK")
