# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=hot
# pyre-check: spec-folds=kwonly_defaults_inline
# Self-checking guard for the pins under an inlined callee's keyword-only
# defaults.
#
# Seeding those locals no longer probes `__kwdefaults__` per call.  The mapping
# a definition builds carries a version, so the walker bakes each entry's cell,
# records a quasi-immutable marker on `Function.w_kw_defs` and one on the
# mapping's strategy version, drains both with a single `GuardNotInvalidated`,
# and reads the cell's field live.  Three separate mechanisms therefore stand
# between a mutation and the next call, and a census can only say the fold
# fired -- not that the pins are honest.  Each site below moves one of them and
# prints a wrong number if that one is missing.
#
# Sites:
#   A  overwrite an existing entry in place.  A cell that already holds an
#      object absorbs the store and bumps no version, so nothing invalidates:
#      this is answered by the live field read alone, and a fold that baked the
#      VALUE rather than the cell keeps returning the old one.
#   B  delete an entry, then add it back.  Both move the strategy version, and
#      the deleted window is observable on its own -- the call must raise
#      rather than seed a stale default.
#   C  rebind the attribute.  `f.__kwdefaults__ = d` stores `d` itself, so only
#      the `Function.w_kw_defs` marker can catch this; the version marker is
#      still watching the old mapping, which no longer answers anything.
#   D  keep calling after that rebind.  A plain dict has no version to pin, so
#      the resolve declines and the callee seeds through the ordinary path --
#      which must still see a later in-place mutation.
N = 20000


def g(x, *, step=1, tag="a"):
    return x + step, tag


def hot(n, want_step, want_tag):
    """Run the loop hot and count answers that disagree with the mapping."""
    bad = 0
    for i in range(n):
        value, tag = g(i)
        if value != i + want_step or tag != want_tag:
            bad += 1
    return bad


def main():
    # Warm first so the sites below mutate a compiled loop rather than an
    # interpreted one.
    if hot(N, 1, "a"):
        print("FAIL warm-up disagreed with the declared defaults")
        return 1

    g.__kwdefaults__["step"] = 5
    bad = hot(N, 5, "a")
    if bad:
        print(f"FAIL site A missed an in-place overwrite on {bad} of {N} calls")
        return 1

    del g.__kwdefaults__["tag"]
    missing = 0
    for i in range(N):
        try:
            g(i)
        except TypeError:
            missing += 1
    if missing != N:
        print(f"FAIL site B seeded a deleted default on {N - missing} of {N} calls")
        return 1

    g.__kwdefaults__["tag"] = "z"
    bad = hot(N, 5, "z")
    if bad:
        print(f"FAIL site B missed the re-added default on {bad} of {N} calls")
        return 1

    g.__kwdefaults__ = {"step": 9, "tag": "q"}
    bad = hot(N, 9, "q")
    if bad:
        print(f"FAIL site C missed the rebound mapping on {bad} of {N} calls")
        return 1

    g.__kwdefaults__["step"] = 11
    bad = hot(N, 11, "q")
    if bad:
        print(f"FAIL site D missed a mutation after the rebind on {bad} of {N} calls")
        return 1

    print("PASS kwdefaults invalidation")
    return 0


import sys

sys.exit(main())
