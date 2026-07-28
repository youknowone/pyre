# pyre-check: max-pypy-ratio=14
# A CALL_ASSEMBLER guard failure whose bridge resume spans two frames: the
# walk runs the resumed region to the callee's return, executing the recursive
# calls inside it concretely, and the walk's result completes the callee. If
# the caller instead rewinds to the guard state and replays through the
# blackhole, every call in the region runs a second time — invisible in the
# returned value (the recursion is pure) but not in CALLS, which counts one
# entry per invocation. Divergence starts after iteration 7000 so the guard
# fails on an already-compiled, already-called-through trace.
MOD = 1000003

memo = {}
CALLS = [0]


def rec_memo(n):
    CALLS[0] += 1
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
    CALLS[0] += 1
    if n <= 1:
        return n
    if n % 2 == 0:
        return (rec_plain(n // 2) * 3 + 7) % MOD
    return (rec_plain(n - 1) + n * 5) % MOD


def main():
    acc = 0
    for i in range(1, 9001):
        n = (i * 37) % 211 + 2
        if i > 7000:
            n = n * 31 + 1
            acc = (acc + rec_plain(n) * 2 + rec_memo(n)) % MOD
        else:
            acc = (acc + rec_memo(n) + rec_plain(n)) % MOD
    print("acc", acc, len(memo))
    print("calls", CALLS[0])


main()
