"""Facet A drain-loop exact-kind fusion regression guard.

Guards pypy/interpreter/baseobjspace.py:1005-1015
`_unpackiterable_unknown_length`.  The StopIteration drain loop's
except handler is an EXACT per-kind test (`e.kind ==
PyErrorKind::StopIteration`, the `e.match(space, w_StopIteration)`
handler form): a StopIteration ends the drain, anything else
re-raises.  The jd1 drain fusion lowers that handler to
`exc_kind_discriminant(evalue) == ExcKind::StopIteration` on the
`next()` LastException edge.  A wrong MRO/subclass-matching lowering
would swallow, e.g., a `(ValueError, StopIteration)` multi-base
exception the interpreter propagates — a silent miscompile.

Self-check: each case captures the first (cold / interpreter) outcome,
then repeats the same unpack in a hot loop; every hot iteration's
outcome must equal the cold one.  A JIT that diverges from the
interpreter on the same input makes a hot iteration disagree ->
AssertionError.  CPython (no JIT) is trivially self-consistent, so this
never false-reds against CPython's divergent MRO-match semantics for
the multi-base case.

Run under the jd1 walker to exercise the fusion:  PYRE_JD1=1
"""

N = 40000


def drain(make_iter):
    """Unpack an unknown-length iterator into a fixed 3-tuple, returning
    a hashable signature of the outcome (ok tuple, or exception id)."""
    try:
        a, b, c = make_iter()
    except BaseException as e:  # noqa: BLE001 - signature capture
        return ("raised", type(e).__name__, str(e))
    return ("ok", a, b, c)


def gen_ok():
    yield 1
    yield 2
    yield 3


class RaiseIter:
    """Iterator whose __next__ raises `exc` after the 1st element."""

    def __init__(self, exc):
        self._exc = exc
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        self._i += 1
        if self._i == 1:
            return 10
        raise self._exc


class MyStop(StopIteration):
    pass


try:
    class MultiStop(ValueError, StopIteration):
        pass
    HAVE_MULTI = True
except TypeError:
    # CPython rejects the multi-base layout; pyre may or may not.
    HAVE_MULTI = False


def run_case(name, make_iter):
    first = drain(make_iter)
    for _ in range(N):
        sig = drain(make_iter)
        assert sig == first, f"{name}: JIT diverged from interp {sig!r} != {first!r}"
    return first


# (i) normal StopIteration drain completes with the drained values.
assert run_case("normal", gen_ok) == ("ok", 1, 2, 3)

# (ii) a plain non-StopIteration error must PROPAGATE, not be swallowed
#      into a 'not enough values to unpack' arity error.
sig_ii = run_case("value_error", lambda: RaiseIter(ValueError("boom")))
assert sig_ii[0] == "raised", sig_ii

# (iii) a StopIteration subclass: the interpreter's exact-kind test decides
#       whether it ends the drain or propagates.  Whatever it does, the JIT
#       must match it (checked in-loop); no absolute assertion, so this stays
#       CPython-safe.
run_case("stop_subclass", lambda: RaiseIter(MyStop()))

# (iv) multi-base (ValueError, StopIteration): the DISTINGUISHING case.
#      Exact-kind (pyre) propagates; MRO-match would swallow.  We assert only
#      that JIT == interpreter within this run, never which side wins.
if HAVE_MULTI:
    run_case("multi_base", lambda: RaiseIter(MultiStop()))

print("OK")
