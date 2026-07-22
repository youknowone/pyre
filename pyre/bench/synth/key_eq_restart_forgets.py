"""A key comparison that disturbs the table is retried from scratch.

`ll_dict_lookup` answers a disturbed probe with `return ll_dict_lookup(...)`
(`rordereddict.py:1057-1062`), so the restarted lookup has no memory of the
comparisons the first attempt already made.  A candidate that answered `False`
before the mutation is therefore compared again, and may answer differently the
second time.

Only one colliding key is stored, so the probe order among equal-hash keys
cannot decide the outcome: the sole question is whether the rejected candidate
is revisited.
"""


class Filler:
    """Distinct hashes, so filling never runs a key comparison of its own."""

    def __init__(self, i):
        self.i = i

    def __hash__(self):
        return 1000 + self.i

    def __eq__(self, other):
        return self is other


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
            # Answer False, but leave state in which a second comparison
            # answers True, and replace the backing arrays so the lookup is
            # entitled to make one.
            self.value = 2
            container = self.owner[0]
            if isinstance(container, dict):
                for i in range(8):
                    container[Filler(i)] = i
            else:
                for i in range(8):
                    container.add(Filler(i))
            return False
        return self.value == other.value


def run_set(operation):
    owner = [None]
    container = set()
    owner[0] = container
    container.add(Key("A", 1, owner))
    probe = Key("P", 2, owner)
    hit = None
    if operation == "add":
        container.add(probe)
    elif operation == "contains":
        hit = probe in container
    else:
        container.discard(probe)
    return sorted(k.tag for k in container if isinstance(k, Key)), hit


def run_dict(operation):
    owner = [None]
    container = {}
    owner[0] = container
    container[Key("A", 1, owner)] = 1
    probe = Key("P", 2, owner)
    hit = None
    if operation == "setitem":
        container[probe] = -1
    elif operation == "getitem":
        hit = container.get(probe, "miss")
    elif operation == "setdefault":
        hit = container.setdefault(probe, -1)
    else:
        hit = container.pop(probe, "miss")
    return sorted(k.tag for k in container if isinstance(k, Key)), hit


def main():
    for operation in ("add", "contains", "discard"):
        print("set", operation, run_set(operation))
    for operation in ("setitem", "getitem", "setdefault", "pop"):
        print("dict", operation, run_dict(operation))

    # Hot loop over a settled container: a plain colliding-key probe, kept
    # allocation-free so the trace exercises the probe path without churning
    # finalizer-bearing storage boxes.  (The restart correctness above runs in
    # the interpreter, where the deferred probe lives regardless of the JIT.)
    settled = {Recorded(i): i for i in range(3)}
    hits = 0
    for _ in range(20000):
        if Recorded(1) in settled:
            hits += 1
    print("hot", hits)


class Recorded:
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return 5

    def __eq__(self, other):
        return isinstance(other, Recorded) and self.value == other.value


main()
