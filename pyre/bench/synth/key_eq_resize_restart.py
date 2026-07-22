"""Key comparisons that grow the container they are probing.

`ll_dict_lookup` restarts the whole lookup when a `keyeq` call replaced the
entries or indexes array, so a bucket scan can be re-entered an unbounded
number of times.  The comparisons here answer the same way every time they
run, so the restarts are invisible in the result while still exercising the
replay: every operation below must report the same counts it would with an
inert `__eq__`.
"""


class Key:
    def __init__(self, value, owner):
        self.value = value
        self.owner = owner

    def __hash__(self):
        return 11

    def __eq__(self, other):
        # Grow the container from inside the probe, once, so the backing
        # arrays are replaced underneath the scan that is still running.
        container = self.owner[0]
        if container is not None and not self.owner[1]:
            self.owner[1] = True
            if isinstance(container, dict):
                for i in range(48):
                    container[("filler", i)] = i
            else:
                for i in range(48):
                    container.add(("filler", i))
        return isinstance(other, Key) and self.value == other.value


def run_set(operation, probe_value):
    owner = [None, False]
    container = set(Key(i, owner) for i in range(6))
    owner[0] = container
    probe = Key(probe_value, owner)
    hit = None
    if operation == "add":
        container.add(probe)
    elif operation == "contains":
        hit = probe in container
    else:
        container.discard(probe)
    return len(container), hit


def run_dict(operation, probe_value):
    owner = [None, False]
    container = {Key(i, owner): i for i in range(6)}
    owner[0] = container
    probe = Key(probe_value, owner)
    hit = None
    if operation == "setitem":
        container[probe] = -1
    elif operation == "getitem":
        hit = container.get(probe)
    elif operation == "delitem":
        try:
            del container[probe]
        except KeyError:
            hit = "keyerror"
    else:
        hit = container.pop(probe, "missing")
    return len(container), hit


def main():
    for value in (3, 99):
        for operation in ("add", "contains", "discard"):
            print("set", operation, value, run_set(operation, value))
        for operation in ("setitem", "getitem", "delitem", "pop"):
            print("dict", operation, value, run_dict(operation, value))

    # Hot loop: once the JIT has traced the colliding probe, the answer must
    # stay stable.  Probe pre-built containers so the loop allocates nothing;
    # an inert owner keeps `__eq__` from growing them, so every iteration runs
    # the same bucket scan without churning the allocator.
    inert = [None, False]
    settled_set = set(Key(i, inert) for i in range(6))
    settled_dict = {Key(i, inert): i for i in range(6)}
    probe = Key(3, inert)
    hits = 0
    for _ in range(20000):
        if probe in settled_set:
            hits += 1
        if settled_dict.get(probe) is not None:
            hits += 1
    print("hot", hits)


main()
