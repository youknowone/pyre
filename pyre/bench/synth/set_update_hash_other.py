# difference_update and intersection_update hash each element of the other
# operand as it is consumed, so an unhashable element raises and a raising
# __hash__ propagates -- including when self is empty and nothing can match.
# Ordinary updates are unchanged. A warmup loop exercises the update path.
def warm(n):
    acc = 0
    for i in range(n):
        s = {0, 1, 2, 3}
        s.difference_update([i % 4])
        s.intersection_update([0, 1, 2, 3])
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


def du(s, o):
    s.difference_update(o)
    return sorted(s, key=repr)


def iu(s, o):
    s.intersection_update(o)
    return sorted(s, key=repr)


def sdu(s, o):
    s.symmetric_difference_update(o)
    return sorted(s, key=repr)


def main():
    print("warm", warm(15000))
    # the other operand is hashed as it is consumed
    m("iu_unhashable", lambda: iu({1}, [[]]))
    m("du_raising", lambda: du({1}, [RaisingHash()]))
    m("iu_raising", lambda: iu({1}, [RaisingHash()]))
    # ... even when self is empty and there is nothing to compare against
    m("iu_empty_self_unhashable", lambda: iu(set(), [[]]))
    m("iu_empty_self_raising", lambda: iu(set(), [RaisingHash()]))
    # ordinary updates are unchanged
    m("du_ok", lambda: du({1, 2, 3}, [2]))
    m("iu_ok", lambda: iu({1, 2, 3}, [2, 3, 4]))
    m("sdu_ok", lambda: sdu({1, 2}, [2, 3]))
    m("iu_multi", lambda: iu({1, 2, 3}, [1, 2]))
    m("du_str", lambda: du(set("abc"), "b"))
    m("iu_empty_other", lambda: iu({1, 2}, []))
    m("du_empty_other", lambda: du({1, 2}, []))
    m("du_int_float", lambda: du({1, 2}, [1.0]))
    m("iu_two_others", lambda: iu({1, 2}, [1]))
    m("du_two_others_unhashable", lambda: du({1}, [1]))


main()
