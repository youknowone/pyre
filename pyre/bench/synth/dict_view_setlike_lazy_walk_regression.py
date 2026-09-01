# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=warm
# Self-checking regression guard for the two `SetLikeDictView` walks that step
# their operand one item at a time.
#
# `dictmultiobject.py descr_isdisjoint` opens `space.iter(w_other)` and tests
# `space.contains_w(self, w_item)` per step, so it answers `False` at the first
# common element and never asks the iterable for the rest.  Materialising the
# operand first instead reaches items the walk had already decided against: an
# iterable that yields a common element and then raises propagated that
# exception rather than answering, and a generator was drained past the answer.
#
# `dictmultiobject.py _all_contained_in` walks `space.iter(w_dictview)` for the
# same reason on the other side.  A `contains_w` that mutates the dict behind
# the view is caught by the view iterator's own size check on the next step; a
# snapshot taken before the loop answers from the pre-mutation copy, so the
# mutation went unreported and the comparison returned an answer at all.
#
# A set answers `NotImplemented` for a view, so `set OP view` is decided by the
# reflected call on the view with the operator flipped -- the same two walks.
# Reducing both sides to sets instead answers those from a snapshot as well,
# which is why the reflected forms are asserted here beside the direct ones.
#
# Every expectation below is the value CPython 3.14 and PyPy both produce.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

failures = []


class Consumed:
    n = 0


def counted(items):
    for item in items:
        Consumed.n += 1
        yield item


def raising_after(items):
    for item in items:
        yield item
    raise ValueError("the walk should have stopped before here")


class MutatesOnEq:
    """Grows `victim` from inside the comparison the containment test runs."""

    def __init__(self, i):
        self.i = i

    def __hash__(self):
        return hash(self.i)

    def __eq__(self, other):
        victim[len(victim) + 100] = "grown"
        return isinstance(other, MutatesOnEq) and other.i == self.i


victim = {}


def check(label, fn, expected):
    """Assert `fn()` answers `expected`, or raises it when it names a type."""
    try:
        got = repr(fn())
    except BaseException as e:  # noqa: BLE001 - the type is the assertion
        got = "!! " + type(e).__name__
    if got != expected:
        failures.append("%s: expected %s, got %s" % (label, expected, got))


def isdisjoint_stops_at_first_hit():
    Consumed.n = 0
    answer = {1: "a"}.keys().isdisjoint(counted([1, 2, 3, 4, 5]))
    return answer, Consumed.n


def isdisjoint_walks_all_when_disjoint():
    Consumed.n = 0
    answer = {9: "a"}.keys().isdisjoint(counted([1, 2, 3]))
    return answer, Consumed.n


def warm(n):
    """The hot loop the JIT has to compile for this guard to cover it."""
    hits = 0
    keys = {1: "a", 2: "b"}
    other = {2: "b", 3: "c"}
    for _ in range(n):
        if not keys.keys().isdisjoint(other.keys()):
            hits += 1
        if keys.keys() == {1, 2}:
            hits += 1
        if {1, 2} >= keys.keys():
            hits += 1
        if keys.keys().isdisjoint((7, 8)):
            hits += 1
    return hits


def lazy_isdisjoint_cases():
    # The operand raises only after the element that decides the answer.
    check(
        "isdisjoint_keys_hit_then_raise",
        lambda: {1: "a", 2: "b"}.keys().isdisjoint(raising_after([1])),
        "False",
    )
    check(
        "isdisjoint_items_hit_then_raise",
        lambda: {1: "a"}.items().isdisjoint(raising_after([(1, "a")])),
        "False",
    )
    # A genuinely disjoint operand has to be walked to the end, so the same
    # iterable does raise there.
    check(
        "isdisjoint_keys_miss_then_raise",
        lambda: {9: "a"}.keys().isdisjoint(raising_after([1])),
        "!! ValueError",
    )
    # How much of the operand the walk consumed.
    check("isdisjoint_stops_at_first_hit", isdisjoint_stops_at_first_hit, "(False, 1)")
    check("isdisjoint_walks_all_when_disjoint", isdisjoint_walks_all_when_disjoint, "(True, 3)")
    # `self is w_other` short-circuits on the length before any walk, so it
    # needs the identical view object, not an equal one.
    same = {1: "a"}.keys()
    check("isdisjoint_identical_view", lambda: same.isdisjoint(same), "False")
    same_empty = {}.keys()
    check("isdisjoint_identical_empty_view", lambda: same_empty.isdisjoint(same_empty), "True")
    check(
        "isdisjoint_equal_but_distinct",
        lambda: {1: "a"}.keys().isdisjoint({1: "a"}.keys()),
        "False",
    )
    # Set-like operands keep answering the same way.
    check("isdisjoint_set_hit", lambda: {1: "a"}.keys().isdisjoint({1, 2}), "False")
    check("isdisjoint_set_miss", lambda: {1: "a"}.keys().isdisjoint({7, 8}), "True")
    check(
        "isdisjoint_big_set_miss",
        lambda: {1: "a"}.keys().isdisjoint(set(range(100, 400))),
        "True",
    )
    check(
        "isdisjoint_big_set_hit",
        lambda: {150: "a"}.keys().isdisjoint(set(range(100, 400))),
        "False",
    )
    # A non-iterable operand still raises from `iter`.
    check("isdisjoint_int", lambda: {1: "a"}.keys().isdisjoint(1), "!! TypeError")


def live_walk_cases():
    global victim

    # The comparison walks the view live, so a mutation from inside the
    # containment test is reported by the iterator's size check.
    victim = {MutatesOnEq(0): 1, MutatesOnEq(1): 2}
    check(
        "eq_mutating_eq",
        lambda: victim.keys() == {MutatesOnEq(0), MutatesOnEq(1)},
        "!! RuntimeError",
    )
    check("eq_mutating_eq_len", lambda: len(victim), "3")
    victim = {MutatesOnEq(0): 1, MutatesOnEq(1): 2}
    check(
        "le_mutating_eq",
        lambda: victim.keys() <= {MutatesOnEq(0), MutatesOnEq(1)},
        "!! RuntimeError",
    )
    # `ge` walks the other operand, so the set is what is iterated and the
    # view's size check is not the one on duty.
    victim = {MutatesOnEq(0): 1}
    check("ge_mutating_eq", lambda: victim.keys() >= {MutatesOnEq(0)}, "True")
    # The same mutation, reached from the set side.
    victim = {MutatesOnEq(0): 1, MutatesOnEq(1): 2}
    check(
        "set_eq_view_mutating_eq",
        lambda: {MutatesOnEq(0), MutatesOnEq(1)} == victim.keys(),
        "!! RuntimeError",
    )
    check("set_eq_view_mutating_eq_len", lambda: len(victim), "3")


def plain_comparison_cases():
    a = {"a": 1, "b": 2}
    b = {"a": 1, "b": 2, "c": 3}
    check("eq_plain", lambda: a.keys() == {"a", "b"}, "True")
    check("lt_plain", lambda: a.keys() < b.keys(), "True")
    check("le_plain", lambda: a.keys() <= b.keys(), "True")
    check("gt_plain", lambda: b.keys() > a.keys(), "True")
    check("ge_plain", lambda: b.keys() >= a.keys(), "True")
    check("ne_plain", lambda: a.keys() != b.keys(), "True")
    check("eq_items", lambda: a.items() == {("a", 1), ("b", 2)}, "True")
    check("eq_not_set_like", lambda: a.keys() == [1, 2], "False")
    # A set answers `NotImplemented` for a view, so every one of these is
    # decided by the reflected call on the view with the operator flipped.
    check("set_eq_view", lambda: {"a", "b"} == a.keys(), "True")
    check("set_ne_view", lambda: {"a", "b"} != a.keys(), "False")
    check("set_lt_view", lambda: {"a"} < a.keys(), "True")
    check("set_lt_view_equal", lambda: {"a", "b"} < a.keys(), "False")
    check("set_le_view", lambda: {"a", "b"} <= a.keys(), "True")
    check("set_le_view_bigger", lambda: {"a", "b", "z"} <= a.keys(), "False")
    check("set_gt_view", lambda: {"a", "b", "c"} > a.keys(), "True")
    check("set_gt_view_equal", lambda: {"a", "b"} > a.keys(), "False")
    check("set_ge_view", lambda: {"a", "b"} >= a.keys(), "True")
    check("set_ge_view_smaller", lambda: {"a"} >= a.keys(), "False")
    check("view_lt_set", lambda: a.keys() < {"a", "b", "c"}, "True")
    check("view_le_set", lambda: a.keys() <= {"a", "b"}, "True")
    check("view_gt_set", lambda: a.keys() > {"a"}, "True")
    check("view_ge_set", lambda: a.keys() >= {"a", "b"}, "True")
    check("frozenset_eq_view", lambda: frozenset({"a", "b"}) == a.keys(), "True")
    check("view_eq_frozenset", lambda: a.keys() == frozenset({"a", "b"}), "True")
    check("set_eq_items_view", lambda: {("a", 1), ("b", 2)} == a.items(), "True")
    # A values view is not set-like, so the comparison never reaches the walk.
    check("set_eq_values_view", lambda: {1, 2} == a.values(), "False")
    check("values_view_eq_set", lambda: a.values() == {1, 2}, "False")


def main():
    lazy_isdisjoint_cases()
    live_walk_cases()
    plain_comparison_cases()
    # The same walks under the JIT, after the loop above has compiled.
    hot = warm(3000)
    if hot != 3000 * 4:
        failures.append("warm() answered %d, expected %d" % (hot, 3000 * 4))
    # The interpreter-level cases again, now that the walks are compiled.
    lazy_isdisjoint_cases()
    live_walk_cases()
    plain_comparison_cases()

    if failures:
        for line in failures:
            print("FAIL", line)
        return 1
    print("PASS dict-view set-like lazy walk")
    return 0


raise SystemExit(main())
