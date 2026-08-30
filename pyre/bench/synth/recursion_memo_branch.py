# pyre-check: spec-folds=binary_op_int,truth_bool,compare_op_int
# The three folds that carry most of the corpus's fold traffic, and nothing
# declared any of them, so switching one off was a silent change. This fixture
# fires them 398/69/65 times, the widest margin of any fixture firing all three.
# Memoized vs plain recursion with post-warm-up branch divergence. The
# memo-dict store (memo[n] = r) once died with a TypeError after warm-up
# (an empty-string type name from a clobbered class read on the dict-store
# path during a deopt-resumed recursive frame). Deterministic; divergence
# (deeper args, branch flip) starts after the loop is compiled.
MOD = 1000003

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

memo = {}


def rec_memo(n):
    if n <= 1:
        return n
    if n in memo:
        return memo[n]
    if n % 2 == 0:
        r = (rec_memo(n // 2) * 3 + 7) % MOD
    else:
        r = (rec_memo(n - 1) + n * 5) % MOD
    memo[n] = r
    return r


def rec_plain(n):
    if n <= 1:
        return n
    if n % 2 == 0:
        return (rec_plain(n // 2) * 3 + 7) % MOD
    return (rec_plain(n - 1) + n * 5) % MOD


def main():
    acc = 0
    for i in range(1, 5001):
        n = (i * 37) % 211 + 2
        if i > 3500:
            n = n * 31 + 1
            acc = (acc + rec_plain(n) * 2 + rec_memo(n)) % MOD
        else:
            acc = (acc + rec_memo(n) + rec_plain(n)) % MOD
        if i % 1000 == 0:
            print("checksum1", i, acc)
    print("final1", acc, len(memo))


main()
