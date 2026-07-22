# A bucket probe that runs a user `__eq__` can have that callback `clear()` the
# very container being probed.  PyPy's set and dict answer such a mid-probe
# clear differently, and the difference is deterministic strategy-layer
# behaviour, not probe-order noise.
#
# A set captures its storage before probing (`d = self.unerase(w_set.sstorage)`,
# setobject.py:942) and `clear` swaps in a fresh empty box
# (`switch_to_empty_strategy`, :922).  The probe therefore finishes against the
# orphaned snapshot: an element it then inserts lands in the dropped box and is
# lost, and a membership test answers as if the clear never happened.
#
# An object dict clears in place — `ll_dict_clear` reallocates `d.entries`
# (rordereddict.py:1360) on the same table — so the `entries != d.entries`
# paranoia arm of `ll_dict_lookup` (:1058) fires and the probe restarts against
# the refilled dict, landing on whatever the callback re-inserted.
N = 4000


class Key:
    """Colliding key whose first `A`-valued comparison clears the container and
    refills it with a replacement that equals the probe."""

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


def build_set():
    owner = [None, None]
    container = set()
    owner[0] = container
    owner[1] = Key("Q", 99, owner)
    container.add(Key("B", 50, owner))
    container.add(Key("A", 1, owner))
    return owner


def run_set(operation):
    owner = build_set()
    container = owner[0]
    probe = Key("P", 99, owner)
    hit = None
    if operation == "add":
        container.add(probe)
    elif operation == "contains":
        hit = probe in container
    else:
        container.discard(probe)
    return sorted(k.tag for k in container), hit


def build_dict():
    owner = [None, None]
    container = {}
    owner[0] = container
    owner[1] = Key("Q", 99, owner)
    container[Key("B", 50, owner)] = 0
    container[Key("A", 1, owner)] = 0
    return owner


def run_dict(operation):
    owner = build_dict()
    container = owner[0]
    probe = Key("P", 99, owner)
    hit = None
    if operation == "setitem":
        container[probe] = -1
    elif operation == "getitem":
        hit = container.get(probe, "miss")
    elif operation == "setdefault":
        hit = container.setdefault(probe, -1)
    else:
        hit = container.pop(probe, "miss")
    return sorted(k.tag for k in container), hit


def hot_loop():
    # Keep the reentrant-clear probe on a hot path so the JIT-compiled residual
    # takes the same restart (dict) / orphaned-snapshot (set) decision as the
    # interpreter.  `Key` has no finalizer, so the churned storage boxes stay
    # off the finalizer path.
    acc = 0
    i = 0
    while i < N:
        _, dict_hit = run_dict("getitem")
        set_tags, _ = run_set("add")
        acc = acc + (0 if dict_hit == "miss" else dict_hit) + len(set_tags)
        i = i + 1
    return acc


def main():
    for operation in ("add", "contains", "discard"):
        print("set", operation, run_set(operation))
    for operation in ("setitem", "getitem", "setdefault", "pop"):
        print("dict", operation, run_dict(operation))
    print("hot", hot_loop())


main()
