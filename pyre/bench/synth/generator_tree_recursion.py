# pyre-check: max-pypy-ratio=7.6
# Generator-driven accumulation over recursive tree/linear results. The
# tree_sum recursion once silently miscompiled on cranelift (first checksum
# already wrong) and recovered a regalloc panic on dynasm. Deterministic;
# after k > 7000 the generator switches to a different recursive mix
# (post-warm-up branch divergence).
#
# `decay` (rlib/jit.py:588, default 40) scales every JitCounter entry down, and
# counter.py:104-121 applies that scaling once per 32 minor collections. How far
# a guard's counter has advanced by the time the workload reaches it therefore
# depends on how much the process has allocated so far, and for this recursion
# that is a per-ISA quantity: x86 spills every CALL_ASSEMBLER result into a fresh
# JITFRAME slot (`x86/assembler.rs:7319`, `:7396` -> `allocate_slot`) where
# aarch64 keeps it in the regalloc result register (`aarch64/assembler.rs:6037`),
# so the two dynasm backends run a different minor-collection schedule over the
# same trace. Left at the default, `guard_failures` reads 2957/2955/2951 across a
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
