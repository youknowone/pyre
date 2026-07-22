"""Cranelift-only GC crash: garbage type id during finalization ordering.

    thread panicked at majit/majit-gc/src/trace.rs:934:
      index out of bounds: the len is 142 but the index is 4294967254
      MiniMarkGC::finalizer_children
      MiniMarkGC::recursively_clear_finalization_ordering
      MiniMarkGC::incremental_mark_step
      MiniMarkGC::do_collect_nursery
      MiniMarkGC::alloc_with_type
      majit_backend_cranelift::compiler::gc_alloc_nursery_shim

`4294967254` is `(u32)-42`: `finalizer_children` (collector.rs:2485) reads a type
id off an object header that is not a live, initialized object.  Sets and dicts
each own a GC storage box carrying a finalizer (it drops the backing Rust
IndexMap), so a hot loop that allocates fresh containers under the cranelift JIT
churns finalizer-bearing nursery boxes; one reaches finalization ordering with a
garbage header.

Intermittent -- CI (macos-26-arm64) tripped it, and locally it is ~2/8 runs, so
run it repeatedly.  Attribution:

    cranelift, JIT on         crashes
    cranelift, PYRE_NO_JIT=1  clean
    dynasm, JIT on            clean

The container code is plain interpreter-side Rust, identical in all three, so the
fault is on the cranelift JIT's nursery allocation path, not the key probe --
this is why the shipped `synth/key_eq_restart_forgets` verifies the restart
answer in a single interpreter pass and keeps its hot loop allocation-free.
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
                for i in range(64):
                    container[Filler(i)] = i
            else:
                for i in range(64):
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

    # Hot loop: the restart path must stay stable once the JIT has traced it.
    total = 0
    for _ in range(20000):
        total += len(run_set("contains")[0]) - len(run_set("discard")[0])
        total += len(run_dict("getitem")[0]) - len(run_dict("pop")[0])
    print("hot", total)


main()
