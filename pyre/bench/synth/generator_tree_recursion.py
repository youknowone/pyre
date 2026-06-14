# Generator-driven accumulation over recursive tree/linear results. The
# tree_sum recursion once silently miscompiled on cranelift (first checksum
# already wrong) and recovered a regalloc panic on dynasm. Deterministic;
# after k > 7000 the generator switches to a different recursive mix
# (post-warm-up branch divergence).
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
