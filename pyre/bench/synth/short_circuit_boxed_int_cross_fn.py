# Cross-function kept-stack short-circuit where a small-int function is traced
# BEFORE a structurally-identical heap-int (>= 256) function.
#
# `x = flag and CONST` keeps the falsy `flag` live across a `goto_if_not`
# guard, so the not-taken arm is unrestorable and the guard must decline to
# the interpreter.  The decline gate read the kept-stack depth through
# `branch_resume_target_stack_depth_any_leg(target, outer_jitcode_index)`, but
# `outer_jitcode_index` is uniformly 0, so for the SECOND distinct function it
# resolved to `jitcodes[0]` (the first function) and read that function's
# metadata at this function's jitcode pc, reporting depth 0 -> the decline was
# skipped and the heap-int arm compiled an unrestorable kept slot, silently
# miscompiling `sc_big` (66732133201 instead of 133333000000).
#
# `sc_small` (const 11) is defined first so it occupies `jitcodes[0]`; `sc_big`
# (const 1000000, a heap int) is the second function and is the one that
# aliased.  Both are bare LOAD_FAST-left / const-right loops so the kept-stack
# guard resumes at depth 1 and the hot loop is the whole trace.  Ordering and
# the heap-magnitude right operand are both load-bearing for the repro.  Pure
# arithmetic -> deterministic checksum.
N = 200000


def sc_small():
    acc = 0
    i = 0
    while i < N:
        flag = i % 3
        x = flag and 11
        acc = acc + x
        i = i + 1
    return acc


def sc_big():
    acc = 0
    i = 0
    while i < N:
        flag = i % 3
        x = flag and 1000000
        acc = acc + x
        i = i + 1
    return acc


def main():
    print(sc_small(), sc_big())


main()
