class Key:
    def __init__(self, value, owner):
        self.value = value
        self.owner = owner

    def __hash__(self):
        return 9

    def __eq__(self, other):
        container = self.owner[0]
        if container is not None and not self.owner[1]:
            self.owner[1] = True
            marker = ("mutated", self.value)
            if isinstance(container, dict):
                container[marker] = 1
            else:
                container.add(marker)
        return isinstance(other, Key) and self.value == other.value


def run_set(operation):
    owner = [None, False]
    container = set(Key(i, owner) for i in range(8))
    owner[0] = container
    probe = Key(1000, owner)
    if operation == "add":
        container.add(probe)
    elif operation == "contains":
        probe in container
    else:
        container.discard(probe)
    return len(container)


def run_dict(operation):
    owner = [None, False]
    container = {Key(i, owner): 1 for i in range(8)}
    owner[0] = container
    probe = Key(1000, owner)
    if operation == "setitem":
        container[probe] = 1
    elif operation == "getitem":
        container.get(probe)
    elif operation == "delitem":
        try:
            del container[probe]
        except KeyError:
            pass
    else:
        container.pop(probe, None)
    return len(container)


for operation in ("add", "contains", "discard"):
    print("set", operation, run_set(operation))
for operation in ("setitem", "getitem", "delitem", "pop"):
    print("dict", operation, run_dict(operation))
