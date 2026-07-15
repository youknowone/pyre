# symmetric_difference_update turns a non-set operand into a set before it
# toggles anything, so the operand is hashed and deduped up front: a duplicate
# toggles once, and a later unhashable element leaves self untouched. update
# consumes its operand element by element instead, so it does mutate before
# raising. A warmup loop exercises the toggle path.
def warm(n):
    acc = 0
    for i in range(n):
        s = {0, 1, 2, 3}
        s.symmetric_difference_update([i % 4, (i + 1) % 4])
        acc += len(s)
    return acc


def m(label, fn):
    try:
        print(label, "->", repr(fn()))
    except BaseException as e:
        print(label, "!!", type(e).__name__, repr(str(e)))


def s(x):
    return sorted(x, key=repr)


def sdu(a, b):
    a.symmetric_difference_update(b)
    return s(a)


def caught(fn, st):
    try:
        fn()
    except TypeError:
        pass
    return s(st)


def main():
    print("warm", warm(15000))
    # a duplicate in the operand toggles once, not twice
    m("sdu_dups", lambda: sdu({1}, [2, 2]))
    m("sdu_dups_present", lambda: sdu({1, 2}, [2, 2]))
    m("keys_xor_dups", lambda: s({1: 0}.keys() ^ [2, 2]))
    m("keys_xor_dups_present", lambda: s({1: 0, 2: 0}.keys() ^ [2, 2]))
    m("sdu_triple", lambda: sdu({1}, [2, 2, 2]))
    # a one-shot operand is consumed once
    m("keys_and_iter", lambda: s({1: 0, 2: 0}.keys() & iter([1, 2])))
    m("keys_sub_iter", lambda: s({1: 0, 2: 0}.keys() - iter([1])))
    m("keys_or_iter", lambda: s({1: 0}.keys() | iter([2])))
    m("keys_and_dups", lambda: s({1: 0, 2: 0}.keys() & [1, 1, 2]))
    m("sdu_iter", lambda: sdu({1}, iter([2])))
    # an unhashable element later in the operand leaves self untouched
    st1 = {1, 2}
    m("sdu_partial", lambda: caught(lambda: st1.symmetric_difference_update([3, []]), st1))
    st2 = {1, 2}
    m("iu_partial", lambda: caught(lambda: st2.intersection_update([1, []]), st2))
    # ... but update adds as it goes, so it mutates before raising
    st3 = {1, 2}
    m("update_partial", lambda: caught(lambda: st3.update([3, []]), st3))
    # ordinary toggles are unchanged
    m("sdu_ok", lambda: sdu({1, 2}, [2, 3]))
    m("sdu_set_rhs", lambda: sdu({1, 2}, {2, 3}))
    m("sdu_empty_rhs", lambda: sdu({1, 2}, []))
    m("sdu_empty_self", lambda: sdu(set(), [1, 2]))
    m("sdu_str", lambda: sdu(set("ab"), "bc"))


main()
