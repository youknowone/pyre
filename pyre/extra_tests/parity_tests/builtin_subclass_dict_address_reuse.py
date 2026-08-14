# CPython-suite gap: attribute tests never allocate enough builtin-subclass
# instances to make one land on the address a dead sibling had.
# parity-tests reason: pyre keeps a builtin-layout instance's `__dict__` and
# weakref lifeline in owner-address-keyed side tables, which CPython and PyPy
# both store as object fields; only pyre can inherit a dead owner's entry.

import weakref


class L(list):
    pass


class S(str):
    pass


def churn_dead(count):
    """Leave `count` dead instances that had attributes and a weakref."""
    for index in range(count):
        value = L([1, 2, 3])
        value.foo = index
        value.bar = "hello"
        weakref.ref(value)


def check_fresh(count):
    """A freshly built instance starts with an empty `__dict__`."""
    inherited = 0
    for index in range(count):
        value = L([1, 2, 3])
        if value.__dict__:
            inherited += 1
        value.foo = index
        if value.__dict__ != {"foo": index}:
            inherited += 1
    return inherited


def check_fresh_str(count):
    inherited = 0
    for index in range(count):
        value = S("x")
        if value.__dict__:
            inherited += 1
        value.tag = index
        if value.__dict__ != {"tag": index}:
            inherited += 1
    return inherited


inherited = 0
for _ in range(40):
    churn_dead(300)
    inherited += check_fresh(300)
    inherited += check_fresh_str(300)

assert inherited == 0, inherited


# A weakref to a dead owner must not be handed to whatever is allocated at its
# address next: the lifeline lives in the same address-keyed table.
alive = []
for _ in range(20):
    churn_dead(300)
    for index in range(300):
        value = L([1, 2, 3])
        reference = weakref.ref(value)
        assert reference() is value, index
        alive.append((value, reference))
    alive.clear()

print("OK")
