# A set's backing storage keys on both the hash and the equality protocol, so
# an element is hashed exactly once per store and a raise from either callback
# aborts that store rather than being swallowed into a corrupt container. An
# element whose __eq__ raises during the bucket probe leaves the container as
# it was: the aborted store appends nothing. A warmup loop exercises the
# element store path with an ordinary hashable.
class Warm:
    def __hash__(self):
        return 3

    def __eq__(self, other):
        return self is other


def warm(n):
    acc = 0
    for i in range(n):
        s = {Warm(), i % 4}
        s.add(i % 3)
        s.update([(i + 1) % 3])
        d = {i % 5: 1}
        acc += len(s) + len(d)
    return acc


def m(label, fn):
    try:
        print(label, "->", repr(fn()))
    except BaseException as e:
        print(label, "!!", type(e).__name__, repr(str(e)))


N = [0]


class Counted:
    """Hashes to a constant, so every store lands in the one bucket."""

    def __hash__(self):
        N[0] += 1
        return 7

    def __eq__(self, other):
        return self is other


def c(label, fn):
    N[0] = 0
    try:
        fn()
        print(label, "->", N[0])
    except BaseException as e:
        print(label, "!!", type(e).__name__, N[0])


class EqBoom:
    """Distinct instances collide, so the probe must reach the raising __eq__."""

    def __hash__(self):
        return 42

    def __eq__(self, other):
        raise ValueError("eq boom")

    def __repr__(self):
        return "EqBoom"


class EqOk:
    def __hash__(self):
        return 42

    def __eq__(self, other):
        return False

    def __repr__(self):
        return "EqOk"


def add_twice(st):
    st.add(EqBoom())
    st.add(EqBoom())
    return len(st)


def setitem_twice(d):
    d[EqBoom()] = 1
    d[EqBoom()] = 2
    return len(d)


def update_boom(st):
    st.update([EqBoom(), EqBoom()])
    return len(st)


def discard_boom(st):
    st.discard(EqBoom())
    return len(st)


def remove_boom(st):
    st.remove(EqBoom())
    return len(st)


def leftover():
    # the raising add must not leave a spurious entry behind
    st = set()
    st.add(EqBoom())
    try:
        st.add(EqBoom())
    except ValueError:
        pass
    return len(st)


def main():
    print("warm", warm(15000))
    # an element is hashed once per store, not once to check and once to place
    c("h_set_ctor", lambda: set([Counted()]))
    c("h_set_add", lambda: set().add(Counted()))
    c("h_set_update", lambda: set().update([Counted()]))
    c("h_frozenset", lambda: frozenset([Counted()]))
    c("h_literal", lambda: {Counted()})
    c("h_comprehension", lambda: {x for x in [Counted()]})
    c("h_dict_fromkeys", lambda: dict.fromkeys([Counted()]))
    c("h_dict_comprehension", lambda: {k: 1 for k in [Counted()]})
    # lookups and the already-single-hash stores are unchanged
    c("h_set_remove", lambda: {1}.remove(Counted()))
    c("h_set_contains", lambda: Counted() in {1})
    c("h_dict_setitem", lambda: {}.__setitem__(Counted(), 1))
    c("h_dict_literal", lambda: {Counted(): 1})
    c("h_dict_update_map", lambda: {}.update({Counted(): 1}))
    c("h_dict_get", lambda: {}.get(Counted()))
    c("h_dict_in", lambda: Counted() in {})
    # an __eq__ raising during the probe propagates out of the store
    m("set_ctor_eq", lambda: len(set([EqBoom(), EqBoom()])))
    m("set_add_eq", lambda: add_twice(set()))
    m("set_update_eq", lambda: update_boom(set()))
    m("frozenset_eq", lambda: len(frozenset([EqBoom(), EqBoom()])))
    m("literal_eq", lambda: len({EqBoom(), EqBoom()}))
    m("comprehension_eq", lambda: len({x for x in [EqBoom(), EqBoom()]}))
    m("dict_fromkeys_eq", lambda: len(dict.fromkeys([EqBoom(), EqBoom()])))
    m("dict_comprehension_eq", lambda: len({k: 1 for k in [EqBoom(), EqBoom()]}))
    m("dict_pairs_eq", lambda: len(dict([(EqBoom(), 1), (EqBoom(), 2)])))
    m("dict_literal_eq", lambda: len({EqBoom(): 1, EqBoom(): 2}))
    m("dict_setitem_eq", lambda: setitem_twice({}))
    # and out of a lookup
    m("contains_eq", lambda: EqBoom() in {EqBoom()})
    m("discard_eq", lambda: discard_boom({EqBoom()}))
    m("remove_eq", lambda: remove_boom({EqBoom()}))
    m("dict_in_eq", lambda: EqBoom() in {EqBoom(): 1})
    # the aborted store leaves the set as it was
    m("leftover_len", leftover)
    # a collision whose __eq__ does not raise still dedupes on identity
    m("collide_ok", lambda: len(set([EqOk(), EqOk()])))
    m("same_obj_twice", lambda: (lambda a: len(set([a, a])))(EqBoom()))


main()
