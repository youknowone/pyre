# Reproducer for a GC root that holds a stale reference after a polymorphic
# slice loop runs in a frame whose globals are not a module's own `__dict__`.
#
# Run it by exec'ing this file's source into a plain dict:
#
#     code = compile(open(__file__).read(), __file__, "exec")
#     exec(code, {"__name__": "__main__"})
#
# Running it directly, or exec'ing it into any module's `__dict__` (including a
# freshly built `types.ModuleType("p").__dict__`), is clean.  So is `PYRE_JIT=0`.
#
# The program completes and prints both lines; the abort happens afterwards,
# when a later collection seeds its roots:
#
#     GC BUG: invalid type_id=... at obj_addr=... site=extra_area
#       obj_in_nursery=true minor_collections=2 major_collections=1
#     ... MiniMarkGC::seed_major_root
#     ... MiniMarkGC::start_incremental_cycle
#     ... MiniMarkGC::do_collect_oldgen_nonmoving
#
# Both ingredients are load-bearing, established by ablation: dropping `lst` and
# `tup` from the loop makes it clean, and replacing the lambda with a
# module-level `def` passed the same way makes it clean.  Neither the slice in
# the callee, the `try`/`except`, nor a raising callee is required --
# `call(lambda: 1)` aborts just as reliably.
N = 50000


class Idx:
    def __init__(self, v):
        self.v = v

    def __index__(self):
        return self.v


def call(fn):
    return fn()


def main():
    seq = "abcdefghij"
    lst = list(range(10))
    tup = tuple(range(10))
    acc = 0
    i = 0
    while i < N:
        a = Idx(i % 5)
        b = Idx(i % 5 + 3)
        acc = acc + len(seq[a:b]) + len(lst[a:b]) + len(tup[a:b])
        i = i + 1
    print("acc", acc)
    print(call(lambda: 1))


main()
