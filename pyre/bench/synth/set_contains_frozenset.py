# A set is unhashable, but when one is used to look an element up it stands
# in for the frozenset holding the same elements, so `in`, discard and remove
# find it. The element is hashed on the way, so a raising __hash__ propagates
# -- membership hashes even against an empty set, while removal from an empty
# set matches nothing without hashing. A warmup loop exercises the membership
# and discard paths.
def warm(n):
    acc = 0
    for i in range(n):
        s = {0, 1, 2, 3}
        if i % 4 in s:
            acc += 1
        s.discard(i % 4)
        acc += len(s)
    return acc


def m(label, fn):
    try:
        print(label, "->", repr(fn()))
    except BaseException as e:
        print(label, "!!", type(e).__name__, repr(str(e)))


class RaisingHash:
    def __hash__(self):
        raise ValueError("nope")


class MySet(set):
    pass


class MyFrozen(frozenset):
    pass


def s(x):
    return sorted(x, key=repr)


def d(st, item):
    st.discard(item)
    return s(st)


def r(st, item):
    st.remove(item)
    return s(st)


def discard_to_empty_readd():
    st = {1}
    st.discard(1)
    st.add(2)
    return s(st)


def main():
    print("warm", warm(15000))
    # a set argument is looked up as the frozenset with the same elements
    m("in_frozen_member", lambda: {1} in {frozenset([1])})
    m("in_frozen_nonmember", lambda: {9} in {frozenset([1])})
    m("in_frozen_empty_rhs", lambda: {1} in set())
    m("in_frozen_in_frozen", lambda: {1} in frozenset([frozenset([1])]))
    m("in_set_in_set_of_sets_miss", lambda: {1} in {frozenset([2])})
    m("empty_set_in_frozen", lambda: set() in {frozenset()})
    m("set_in_frozenset_ctr", lambda: {1} in frozenset([frozenset([1]), 2]))
    m("dunder_contains", lambda: set.__contains__({frozenset([1])}, {1}))
    # ... and so is a set subclass
    m("sub_in_frozen", lambda: MySet([1]) in {frozenset([1])})
    m("sub_in_frozen_miss", lambda: MySet([9]) in {frozenset([1])})
    # a frozenset is hashable, so it never takes the conversion path
    m("frozen_in_frozen", lambda: frozenset([1]) in {frozenset([1])})
    m("myfrozen_in_frozen", lambda: MyFrozen([1]) in {frozenset([1])})
    # discard and remove convert the same way
    m("discard_set_member", lambda: d({frozenset([1]), 2}, {1}))
    m("discard_set_miss", lambda: d({frozenset([1]), 2}, {9}))
    m("remove_set_member", lambda: r({frozenset([1]), 2}, {1}))
    m("remove_set_miss", lambda: r({frozenset([1]), 2}, {9}))
    m("discard_subset", lambda: d({frozenset([1]), 2}, MySet([1])))
    m("remove_subset", lambda: r({frozenset([1]), 2}, MySet([1])))
    # the element is hashed, so a raising __hash__ propagates
    m("in_raising_nonempty", lambda: RaisingHash() in {1})
    m("discard_raising_nonempty", lambda: d({1}, RaisingHash()))
    # membership hashes even when the set is empty and nothing can match
    m("in_raising_empty", lambda: RaisingHash() in set())
    # ordinary membership and removal are unchanged
    m("in_set_of_ints", lambda: 1 in {1, 2})
    m("in_on_frozenset", lambda: 1 in frozenset([1, 2]))
    m("discard_ok", lambda: d({1, 2}, 1))
    m("remove_ok", lambda: r({1, 2}, 1))
    m("remove_missing_int", lambda: r({1, 2}, 9))
    m("discard_missing_int", lambda: d({1, 2}, 9))
    m("discard_to_empty_readd", discard_to_empty_readd)


main()
