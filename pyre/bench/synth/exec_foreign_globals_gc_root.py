# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=main,root:__init__
# Both names are the WORKLOAD text's, not this module's -- the module
# defines nothing from it.  `root:__init__` is measured and belongs to the
# declaration for the same reason the comment below says not to tidy the
# workload: the shapes that reach the JIT are what carry a `ConstRef` into the
# constant pool this fixture guards.
# Guard for the jitcode constant pool's GC rooting.  `constants_r` is a bare
# `Vec<i64>` carrying no GC header, and a jitcode is built per CodeObject at run
# time, so a trace's `Operand::ConstRef` reaches the pool straight off the heap.
# A walk that iterates the pool by value, or that returns early for a minor
# collection, leaves a slot naming an object that has already moved:
#
#     GC BUG: invalid type_id=... obj_in_nursery=true site=extra_area
#     ... MiniMarkGC::seed_major_root
#
# The program finishes and prints first; the abort lands in a later collection
# that seeds its roots, so the signal here is the exit code, not the output.
#
# WHY THE WORKLOAD IS TEXT AND THIS MODULE DEFINES NOTHING FROM IT.  The loop's
# frame has to run under a globals mapping that is not a module's own
# `__dict__`: running the same source directly, or exec'ing it into any module
# dict, is clean.  A module that execs the workload but also defines its names
# is clean too -- two spellings of "re-exec my own source" measured 0 aborts in
# 3 runs each on a binary that aborts through this shape.  So the driver has to
# carry the workload as text and define none of it.
#
# DO NOT TIDY THE WORKLOAD.  Its post-loop statements decide whether this
# reproduces at all.  Measured on a pre-fix `pyre-dynasm`, 6 runs each:
#
#     as written                                       6/6 abort
#     `print("acc", acc)` deleted outright             6/6 abort
#     that print replaced by `assert acc == 450000`    0/6
#     `print(call(lambda: 1))` deleted                 0/6
#
# So the lambda built inside the frame and called is required -- a module-level
# `def` passed the same way is clean -- and an `assert` reading the accumulator
# after the loop silences the abort even though deleting the same line does not.
# The polymorphic slice loop is required: dropping `lst` and `tup` is clean.
#
# `PYRE_JIT=0` is clean, which is the control that makes this the JIT's fixture.
#
# The globals dict deliberately carries only `__name__`, and the compile takes a
# literal filename: the wasm guest defines no `__file__`, and naming it here
# raised `NameError` there while both native backends passed.  Neither is part
# of what this reproduces -- the discriminator is that the mapping is not a
# module's own `__dict__` -- and a mapping holding `__name__` alone is one of
# the spellings that was measured to abort.

WORKLOAD = '''N = 50000


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
'''

exec(compile(WORKLOAD, "<workload>", "exec"), {"__name__": "__main__"})
print("PASS")
