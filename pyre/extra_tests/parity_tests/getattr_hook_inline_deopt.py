# CPython-suite gap: the suite exercises __getattr__ semantics but never runs a
# hooked access hot enough to be compiled, so nothing covers the compiled form.
# parity-tests reason: this targets the pyre-specific guards a compiled
# __getattr__ hook rests on.

"""A compiled `__getattr__` hook answers to the two pins that admitted it.

`objspace.py:710 get_and_call_function` reaches the hook only after the
attribute resolves nowhere, so the compiled form pins the receiver's type
version tag (the type keeps lacking the name, and keeps this hook) and the
instance map (the receiver keeps lacking the name).  Each loop below runs long
enough to be compiled and then invalidates exactly one of those pins mid-loop:
the values recorded before and after must differ at the iteration the pin was
broken, which is what proves the guard deopts rather than the compiled answer
being reused.

The AttributeError case is here for the same reason: a hook that raises for an
unknown name is an ordinary outcome of an inlined body, not a shape the fold may
quietly turn into a returned value.
"""

N = 40000
SWAP = N // 2


class Instance:
    def __getattr__(self, name):
        return "hook:" + name


class Hooked:
    @classmethod
    def __getattr__(cls, name):
        return "cm:%s:%s" % (cls.__name__, name)


class Static:
    @staticmethod
    def __getattr__(name):
        return "sm:" + name


class Raiser:
    def __getattr__(self, name):
        if name == "absent":
            raise AttributeError("no " + name)
        return "ok:" + name


class Installer:
    def __getattr__(self, name):
        # The hook itself gives the instance the attribute, so every later
        # access must read the instance rather than hook again.
        self.installed = "real"
        return "hook:" + name


def instance_shadow():
    """A store during the loop puts the name on the instance."""
    obj = Instance()
    seen = []
    i = 0
    while i < N:
        seen.append(obj.later)
        if i == SWAP:
            obj.later = "instance"
        i += 1
    assert seen[0] == "hook:later", seen[0]
    assert seen[SWAP] == "hook:later", seen[SWAP]
    assert seen[SWAP + 1] == "instance", seen[SWAP + 1]
    assert seen[-1] == "instance", seen[-1]


def hook_replaced():
    """Reassigning `__getattr__` bumps the type's version tag."""

    class Swapped(Hooked):
        pass

    obj = Swapped()
    seen = []
    i = 0
    while i < N:
        seen.append(obj.zed)
        if i == SWAP:
            Swapped.__getattr__ = classmethod(lambda cls, name: "replaced")
        i += 1
    assert seen[0] == "cm:Swapped:zed", seen[0]
    assert seen[SWAP + 1] == "replaced", seen[SWAP + 1]


def name_shadowed_on_type():
    """A class attribute added during the loop wins over the hook."""

    class Shadowed(Static):
        pass

    obj = Shadowed()
    seen = []
    i = 0
    while i < N:
        seen.append(obj.zed)
        if i == SWAP:
            Shadowed.zed = "class"
        i += 1
    assert seen[0] == "sm:zed", seen[0]
    assert seen[SWAP + 1] == "class", seen[SWAP + 1]


def bound_argument_follows_the_receiver_type():
    """A classmethod hook binds the receiver's own class, not the base."""

    class Sub(Hooked):
        pass

    base = Hooked()
    sub = Sub()
    i = 0
    while i < N:
        assert base.q == "cm:Hooked:q"
        assert sub.q == "cm:Sub:q"
        i += 1


def hook_raises():
    """An AttributeError out of the hook reaches the caller every iteration."""
    obj = Raiser()
    hits = 0
    misses = 0
    i = 0
    while i < N:
        hits += len(obj.present)
        try:
            obj.absent
        except AttributeError as exc:
            assert str(exc) == "no absent", exc
            misses += 1
        i += 1
    assert hits == N * len("ok:present"), hits
    assert misses == N, misses


def hook_installs_the_attribute():
    obj = Installer()
    seen = []
    i = 0
    while i < N:
        seen.append(obj.installed)
        i += 1
    assert seen[0] == "hook:installed", seen[0]
    assert seen[1] == "real", seen[1]
    assert seen[-1] == "real", seen[-1]


def main():
    instance_shadow()
    hook_replaced()
    name_shadowed_on_type()
    bound_argument_follows_the_receiver_type()
    hook_raises()
    hook_installs_the_attribute()
    print("OK")


main()
