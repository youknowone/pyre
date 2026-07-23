"""Regression repro: garbage type id during finalization ordering (FIXED).

    thread panicked at majit/majit-gc/src/trace.rs:
      index out of bounds: the len is 142 but the index is 4294967254
      MiniMarkGC::finalizer_children
      MiniMarkGC::recursively_clear_finalization_ordering
      MiniMarkGC::incremental_mark_step
      MiniMarkGC::do_collect_nursery
      MiniMarkGC::alloc_with_type
      majit_backend_cranelift::compiler::gc_alloc_nursery_shim

`4294967254` is `(u32)-42`, the low half of the header FORWARDED_MARKER:
`finalizer_children` was dereferencing an already-moved object.

Root cause (confirmed by deterministic bisection): the generator resume paths
skipped `generator.py` `_invoke_execute_frame`'s
`finally: frame.f_backref = jit.vref_None`,
so the exhausted genexp that `sorted(...)` below fully consumes kept its last
resumer's dead frame reachable through the finalizer graph (the genexp carries
a finalizer via its exception table).  A later minor collection that moved one
of that dead frame's young locals -- the container itself, forwarded through
the live owner while the dead frame is neither a root nor remembered -- left
the frame slot stale, and the major finalization-ordering walk read the
forwarding marker as a type id.  The cranelift-only attribution was a
promotion-timing artifact, not a distinct JIT store bug.  Fixed by the
`generator_invoke_execute_frame` port in baseobjspace.rs, whose `finally`
drops `f_backref` (and `generator_frame_is_finished` clears finished frames).

The natural phases below are intermittent (CI macos-26-arm64 tripped it, ~2/8
locally).  The `run_del_probe` phase reproduces the pre-fix crash
deterministically: an object with `__del__` and a fully-consumed genexp both
sit on the finalizer queue holding the same container.  Build with the pyrex
`gc_stress` feature (see the commented passthrough in pyrex/Cargo.toml) and
run under `MAJIT_GC_STRESS=1` -- every allocation then forces a full
collection and the pre-fix binary panics on the first finalization walk.
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


class Owner:
    """Has __del__, so it joins the FinalizerQueue; owns the probed container."""

    def __init__(self):
        self.box = [None]
        self.container = None

    def __del__(self):
        pass


def run_del_probe(kind):
    o = Owner()
    if kind == "set":
        c = set()
        o.container = c
        o.box[0] = c
        c.add(Key("A", 1, o.box))
        probe = Key("P", 2, o.box)
        _ = probe in c
    else:
        c = {}
        o.container = c
        o.box[0] = c
        c[Key("A", 1, o.box)] = 1
        probe = Key("P", 2, o.box)
        _ = c.get(probe, None)
    # `o` (with __del__) dies at return; the genexp the sum consumes is also
    # queued as a finalizer, and pre-fix its frame still linked the resumer.
    return sum(1 for k in c if isinstance(k, Key))


def main():
    # Deterministic phase first (see docstring): panics the pre-fix binary
    # under MAJIT_GC_STRESS=1 with the gc_stress feature built in.  Runs
    # before the hot loop because every phase crawls under GC stress.
    total = 0
    for i in range(5000):
        total += run_del_probe("set" if i % 2 == 0 else "dict")
    print("del_probe", total)

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
