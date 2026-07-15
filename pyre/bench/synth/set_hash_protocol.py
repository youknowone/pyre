# Adding an element to a set hashes it through the protocol, so an unhashable
# element raises instead of being stored under a structural hash and a raising
# __hash__ propagates. This covers every ingestion path: the set and frozenset
# constructors, set.add, set.update, symmetric_difference_update, and the
# set-literal / set-comprehension opcodes. A warmup loop exercises the
# ordinary add path.
def warm(n):
    acc = 0
    for i in range(n):
        s = set()
        s.add(i % 8)
        s.add((i + 1) % 8)
        s.update([(i + 2) % 8])
        acc += len(s) | len({j for j in range(3)})
    return acc


def m(label, fn):
    try:
        print(label, "->", repr(fn()))
    except BaseException as e:
        print(label, "!!", type(e).__name__, repr(str(e)))


class RaisingHash:
    def __hash__(self):
        raise ValueError("nope")


def build_literal(x):
    return {x}


def build_comprehension(xs):
    return {y for y in xs}


def do_add(x):
    s = set()
    s.add(x)
    return s


def do_update(x):
    s = set()
    s.update([x])
    return s


def do_symdiff_update(x):
    s = set()
    s.symmetric_difference_update([x])
    return s


def main():
    print("warm", warm(15000))
    # a raising __hash__ propagates from every ingestion path
    m("ctor_raising_hash", lambda: set([RaisingHash()]))
    m("frozenset_raising_hash", lambda: frozenset([RaisingHash()]))
    m("add_raising_hash", lambda: do_add(RaisingHash()))
    m("update_raising_hash", lambda: do_update(RaisingHash()))
    m("symdiff_update_raising_hash", lambda: do_symdiff_update(RaisingHash()))
    m("literal_raising_hash", lambda: build_literal(RaisingHash()))
    m("comprehension_raising_hash", lambda: build_comprehension([RaisingHash()]))
    # the element is hashed even when it is not the first one
    m("ctor_raising_hash_2nd", lambda: set([1, RaisingHash()]))
    m("update_raising_hash_2nd", lambda: do_update_two())
    # ordinary sets still build
    m("ctor_ok", lambda: sorted(set([1, 2, 2, 3])))
    m("frozenset_ok", lambda: sorted(frozenset([1, 2])))
    m("add_ok", lambda: sorted(do_add(5)))
    m("update_ok", lambda: sorted(do_update(7)))
    m("symdiff_update_ok", lambda: sorted(do_symdiff_update(9)))
    m("literal_ok", lambda: sorted(build_literal(3)))
    m("comprehension_ok", lambda: sorted(build_comprehension([1, 2, 2])))
    m("ctor_str", lambda: sorted(set("aab")))
    m("ctor_empty", lambda: sorted(set()))
    m("ctor_generator", lambda: sorted(set(i for i in range(3))))
    m("nested_frozenset_ok", lambda: sorted(set([frozenset([1]), frozenset([1])]), key=repr))


def do_update_two():
    s = set()
    s.update([1, RaisingHash()])
    return s


main()
