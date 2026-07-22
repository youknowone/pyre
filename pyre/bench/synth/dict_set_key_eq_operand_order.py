# A dict/set bucket probe compares the key already stored against the incoming
# one, in that order (`ll_dict_lookup` runs `keyeq(checkingkey, key)`).  The
# order decides whose `__eq__` the comparison protocol reaches first, so it is
# observable for any key whose `__eq__` is asymmetric or records that it ran.
N = 20000


class Recorded:
    """Colliding key that appends the operand pair of every comparison."""

    def __init__(self, v, log):
        self.v = v
        self.log = log

    def __hash__(self):
        return 9

    def __eq__(self, other):
        self.log.append((self.v, getattr(other, "v", None)))
        return isinstance(other, Recorded) and other.v == self.v


class LeftOnly:
    """Equal only when it is the left operand, so a swapped probe answers
    differently instead of merely reordering the calls."""

    def __init__(self, v):
        self.v = v

    def __hash__(self):
        return 11

    def __eq__(self, other):
        return isinstance(other, LeftOnly) and other.v == self.v + 1

    def __ne__(self, other):
        return not self.__eq__(other)


def probe_order():
    log = []
    stored = Recorded(1, log)
    s = {stored}
    s.add(Recorded(2, log))
    set_add = list(log)

    log.clear()
    d = {Recorded(1, log): "a"}
    d[Recorded(2, log)] = "b"
    dict_setitem = list(log)

    log.clear()
    Recorded(1, log) in {Recorded(3, log)}
    set_contains = list(log)

    log.clear()
    {Recorded(3, log): 0}.get(Recorded(1, log))
    dict_getitem = list(log)

    return set_add, dict_setitem, set_contains, dict_getitem


def asymmetric():
    # `LeftOnly(0)` is stored; probing with `LeftOnly(1)` hits the same bucket.
    # `stored == probe` is True and `probe == stored` is False, so membership
    # answers differently depending on which side the probe puts first.
    s = {LeftOnly(0)}
    d = {LeftOnly(0): "x"}
    return (
        LeftOnly(1) in s,
        LeftOnly(-1) in s,
        d.get(LeftOnly(1), "miss"),
        d.get(LeftOnly(-1), "miss"),
    )


def hot_loop():
    # Keep the probes on a hot path so the JIT-compiled residual takes the
    # same comparison order as the interpreter.
    acc = 0
    i = 0
    log = []
    while i < N:
        s = {Recorded(i, log)}
        s.add(Recorded(i, log))
        acc = acc + len(s)
        del log[:]
        i = i + 1
    return acc


def main():
    set_add, dict_setitem, set_contains, dict_getitem = probe_order()
    print("set_add", set_add)
    print("dict_setitem", dict_setitem)
    print("set_contains", set_contains)
    print("dict_getitem", dict_getitem)
    print("asymmetric", asymmetric())
    print("hot", hot_loop())


main()
