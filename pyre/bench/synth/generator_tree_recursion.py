# pyre-check: max-pypy-ratio=23
# pyre-check: jitstats-band=guard_failures=8
# Successful bridge closure and a pre-trace Decline are not aborts in
# `MetaInterp._interpret`. Charging both to pyre's local abort ceiling held this
# fixture at 29 bridges / 3600 guard failures; the corrected lifecycle reaches
# 33 / 4382. The PyPy oracle compiles still more (65 bridges) with forcings=0,
# virtualizables forced=0 and nvirtuals=721, so the higher count is coverage,
# not a regression to suppress. Once the manual arithmetic folds were retired
# onto generated interpreter descents, branch runs 33692288311, 33813140363 and
# 33860926996 measured cranelift at 13.6x..19.7x and dynasm at 10.2x..13.8x
# while the exact 3-loop / 33-bridge / 4382-guard shape stayed fixed. 23x is the
# cross-host high plus 15%. The recovery target is PyPy's canonical codewriter
# inline call to the arithmetic body and its zero-forcing per-`MIFrame`
# recursive-frame/blackhole path, not either retired shortcut.
# Jitcounter decay is 0.96 every 32 minor collections
# (majit-trace/src/counter.rs), so guard_failures tracks collection count during
# each guard's warm-up rather than a compile decision. One host measured
# 3648..3661 across nursery sizes before the lifecycle fix; decay=0 now pins
# 4382 everywhere, while loops_compiled=3 and bridges_compiled=33 remain
# gated exactly. The fixture sets decay=0 itself, so the band covers the pinned
# run, not that 13-wide unpinned spread; width 8 is margin (0.22%). Real
# regressions this gate caught moved by hundreds to thousands (828 -> 4923,
# 404 -> 812, 937 -> 7408).
# Generator-driven accumulation over recursive tree/linear results. The
# tree_sum recursion once silently miscompiled on cranelift (first checksum
# already wrong) and recovered a regalloc panic on dynasm. Deterministic;
# after k > 7000 the generator switches to a different recursive mix
# (post-warm-up branch divergence).
#
# `decay` (rlib/jit.py:588, default 40) scales every JitCounter entry down, and
# counter.py:104-121 applies that scaling once per 32 minor collections. How far
# a guard's counter has advanced by the time the workload reaches it therefore
# depends on how much the process has allocated so far, which is why the pin
# exists: anything that shifts allocation volume shifts every counter. Both
# dynasm backends now deliver a CALL_ASSEMBLER or nursery result into the
# regalloc result register rather than a JitFrame slot
# (`move_call_assembler_result`, and `consider_call_malloc_nursery`'s
# `force_allocate_reg`), so neither grows the frame per call the way x86 did.
# Left at the default, `guard_failures` reads 3661/3648/3648 across a
# 1MB/4MB/16MB nursery sweep on one binary with `loops_compiled` and
# `bridges_compiled` unchanged; at 0 the same sweep reads one number.
#
# CPython (the oracle) has no `pypyjit`; PyPy and pyre do. Guarding the import
# keeps the output identical across all three while the param only binds where a
# JIT exists. `set_param` rather than an environment variable because the wasm
# guest sees no environment.
try:
    import pypyjit

    pypyjit.set_param("decay=0")
except ImportError:
    pass

MOD = 1000003


def tree_sum(n):
    if n <= 1:
        return n + 1
    if n % 2 == 0:
        return (tree_sum(n // 2) * 2 + tree_sum(n // 2 - 1)) % MOD
    return (tree_sum((n - 1) // 2) + n) % MOD


def deep(n):
    if n <= 0:
        return 3
    return (deep(n - 1) + n * n) % MOD


def gen_values(limit):
    k = 1
    while k <= limit:
        if k > 7000:
            n = (k * 11) % 523 + 2
            yield (tree_sum(n) * 3 + deep(n % 67)) % MOD
        else:
            n = (k * 7) % 311 + 2
            yield (tree_sum(n) + k) % MOD
        k += 1


def main():
    acc = 0
    cnt = 0
    for v in gen_values(9000):
        acc = (acc + v) % MOD
        cnt += 1
        if cnt % 1800 == 0:
            print("checksum3", cnt, acc)
    print("final3", acc, cnt)


main()
