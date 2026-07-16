# Acceptance test for the branchy-inlined-callee bridge gap: a guard failure
# inside an inlined callee should compile a bridge instead of deopting to the
# interpreter on every crossing.
#
# It mirrors inline_helper.py's add/mul/square/compute call chain, but compute()
# here branches on data. inline_helper.py keeps straight-line callees on purpose
# (it is a passing gated bench, and its callees never branch), so the corpus
# otherwise never exercises a branch inside an inlined callee — the one shape
# that still fails. This fills that coverage hole.
#
# NOT wired into check.py: it FAILS today (~150x) and will until the gap closes.
# Wiring it into a gate now would turn check.py red. Run it directly:
#
#     python pyre/bench/bridge_branchy_callee_regression.py       # PASS (interpreter)
#     target/release/pyre-cranelift pyre/bench/bridge_branchy_callee_regression.py
#
# The gap: the straight compute chain inlines and runs compiled; the moment
# compute() branches, the rare arm's guard failure compiles no bridge and every
# crossing deopts to the blackhole (bridges_compiled=0, loops_aborted grows). The
# framestack-walk path does compile the bridge but is correctness-buggy for this
# shape and stays gated off behind PYRE_P2_DRAIN=0; the drain that replaced it is
# a blackhole safety floor that does not compile. Closing the gap means
# reconstructing the callee frame orthodoxly, not flipping that default.
#
# Structural JIT stats do NOT catch this class reliably: a declined green key can
# read bridges_compiled>=1 with a low abort count yet still run ~200x slow,
# because a decline stops the retry that would raise the count. The signal must
# be a wall-clock ratio.
#
# That ratio is taken between two loops in THIS process, so concurrent load slows
# both alike and the ratio survives a busy machine; an absolute timing gate does
# not. The two drivers are duplicated rather than sharing one parameterized
# driver, which would make the call site polymorphic and guard on the callee
# itself — a different guard than the one under test.

import time

N = 2_000_000
M = 1_000_000_007
SLOWDOWN_LIMIT = 20.0  # the branchy chain may be at most 20x the straight one


def add(a, b):
    return a + b


def mul(a, b):
    return a * b


def square(x):
    return mul(x, x)


def compute_straight(x):
    return add(square(x), x)


def compute_branch(x):
    if x % 7 == 0:
        return add(x, x)
    return add(square(x), x)


def drive_straight(n):
    s = 0
    i = 0
    while i < n:
        s = add(s, compute_straight(i)) % M
        i = add(i, 1)
    return s


def drive_branch(n):
    s = 0
    i = 0
    while i < n:
        s = add(s, compute_branch(i)) % M
        i = add(i, 1)
    return s


def per_iter_ns(drive):
    drive(N)  # compile
    t0 = time.perf_counter()
    total = drive(N)
    return (time.perf_counter() - t0) / N * 1e9, total


base_ns, base_sum = per_iter_ns(drive_straight)
branch_ns, branch_sum = per_iter_ns(drive_branch)

# Closed forms, so the check itself stays cheap. The straight arm adds i*i+i
# each iteration; the branch arm replaces that with 2*i on multiples of 7.
S1 = N * (N - 1) // 2
S2 = (N - 1) * N * (2 * N - 1) // 6
m = (N + 6) // 7  # count of i in [0, N) with i % 7 == 0
S1m = m * (m - 1) // 2
S2m = (m - 1) * m * (2 * m - 1) // 6
expected_straight = (S2 + S1) % M
expected_branch = (S2 + S1 + 7 * S1m - 49 * S2m) % M
if base_sum != expected_straight or branch_sum != expected_branch:
    print(f"FAIL wrong answer: straight={base_sum} branch={branch_sum}")
    raise SystemExit(1)

ratio = branch_ns / base_ns if base_ns > 0 else float("inf")
print(f"straight={base_ns:.2f}ns branch={branch_ns:.2f}ns ratio={ratio:.1f}x")
if ratio > SLOWDOWN_LIMIT:
    print(f"FAIL no bridge for the branch arm: branchy chain {ratio:.0f}x slower")
    raise SystemExit(1)
print("PASS bridge compiled for the inlined callee's branch arm")
