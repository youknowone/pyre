# Companion to dict_set_clear_in_eq_restart.py: the derived set operations and
# dict.update under a mid-probe `clear()` from a key's `__eq__`.  Each strategy
# method captures its operand tables up front (`_symmetric_difference_unwrapped`
# unerases both sides, `rev_update1_dict_dict` walks the source through the
# guard-free low-level dict iterator), so a clear mid-operation orphans the
# cleared side instead of steering the walk or the probes onto the replacement
# storage — and dict.update never raises the application-level changed-size
# error.  Kept out of the parity glob because CPython and PyPy disagree on
# these outcomes (check.py BASEFAILs any CPython/PyPy mismatch).
class Key:
    def __init__(self, tag, value, owner):
        self.tag = tag
        self.value = value
        self.owner = owner

    def __hash__(self):
        return 7

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Key):
            return NotImplemented
        if self.tag == "A" and self.value == 1:
            self.value = 2
            container, replacement = self.owner[0], self.owner[1]
            if isinstance(container, dict):
                container.clear()
                container[replacement] = 0
                container[self] = 0
            else:
                container.clear()
                container.add(replacement)
                container.add(self)
            return False
        return self.value == other.value


def tags(c):
    return sorted(k.tag for k in c)


def attempt(fn):
    try:
        return fn()
    except BaseException as e:
        return type(e).__name__


def build_sets(clear_target):
    owner = [None, None]
    dst = set()
    src = set()
    owner[0] = dst if clear_target == "dst" else src
    owner[1] = Key("Q", 90, owner)
    dst.add(Key("B", 50, owner))
    dst.add(Key("A", 1, owner))
    src.add(Key("P1", 99, owner))
    src.add(Key("P2", 77, owner))
    src.add(Key("P3", 55, owner))
    return owner, dst, src


def set_op(name, op, clear_target):
    owner, dst, src = build_sets(clear_target)
    r = attempt(lambda: op(dst, src))
    extra = tags(r) if isinstance(r, (set, frozenset)) else r
    print("set", name, clear_target, tags(dst), tags(src), extra)


def build_dicts(clear_target):
    owner = [None, None]
    dst = {}
    src = {}
    owner[0] = dst if clear_target == "dst" else src
    owner[1] = Key("Q", 90, owner)
    dst[Key("B", 50, owner)] = 1
    dst[Key("A", 1, owner)] = 2
    src[Key("P1", 99, owner)] = 3
    src[Key("P2", 77, owner)] = 4
    src[Key("P3", 55, owner)] = 5
    return owner, dst, src


def dict_update(clear_target):
    owner, dst, src = build_dicts(clear_target)
    r = attempt(lambda: dst.update(src))
    print("dict update", clear_target, tags(dst), tags(src), r)


def set_remove():
    owner = [None, None]
    c = set()
    owner[0] = c
    owner[1] = Key("Q", 99, owner)
    c.add(Key("B", 50, owner))
    c.add(Key("A", 1, owner))
    r = attempt(lambda: c.remove(Key("P", 99, owner)))
    print("set remove", tags(c), r)


def main():
    for target in ("dst", "src"):
        set_op("difference_update", lambda d, s: d.difference_update(s), target)
        set_op("intersection_update", lambda d, s: d.intersection_update(s), target)
        set_op("symmetric_difference_update",
               lambda d, s: d.symmetric_difference_update(s), target)
        set_op("union", lambda d, s: d | s, target)
        set_op("intersection", lambda d, s: d & s, target)
        set_op("difference", lambda d, s: d - s, target)
        dict_update(target)
    set_remove()


main()
