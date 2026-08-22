# pyre-check: max-pypy-ratio=12
# pyre-check: skip-cpython
# Fitted between the two arms.  With every fold in place the three runners read
# 4.6x / 4.7x on darwin-arm64, 7.2x / 7.6x on ubuntu-24.04 and 9.2x / 10.0x on
# windows -- the spread is pypy's, not ours: these loops are pure fixed cost to
# a JIT that elides them, so its baseline stays near the execution floor and the
# ratio reads the runner as much as the tree.  With
# PYRE_FBW_NO_SPECIALIZE=builtin_fold1,builtin_fold2 putting the same loops back
# on the residual, darwin-arm64 reads 52.9x and 61.4x -- eleven times the folded
# arm on that host.  The gate clears the highest folded reading by 20% and still
# sits an order of magnitude under the residual arm scaled to it.
#
# pyre-check: max-wasm-ratio=10
# pyre-check: spec-folds=builtin_fold1,builtin_fold2
# The wasm ceiling is fitted to the highest reading observed plus 15%:
# ubuntu-24.04 reads 8.2x (wasm 5.66s against dynasm 0.69s) and darwin-arm64
# 8.7x (3.39s against 0.50s).  Two architectures under two load regimes land
# within half a count of each other, so what the ceiling has to clear is
# structural rather than a busy runner.
#
# The structure is the host crossing.  A JIT-emitted trace is its own wasm
# module, so a call leaving it reaches the interpreter through the
# `env.jit_call` trampoline, which marshals func_ptr and arguments through the
# frame call area in shared linear memory and dispatches through the main
# module's indirect function table.  What the fold removes is the frame force,
# the argument rooting, the execution-context resolution and the gateway
# binding; the crossing itself stays, because every fold here still lowers to
# a call into a raw helper.  The tree carries its own contrast: on the same
# ubuntu run `math_folds_hot`, whose folds lower to inline arithmetic instead,
# reads 3.3x.  Every loop below is nothing but folded builtin calls, so the
# crossing is the whole measurement.  The alternative to this allowance is to
# give the trace module direct imports for the raw helpers rather than one
# generic trampoline.
#
# `spec-folds` is the exact instrument neither ratio is.  Six loops sum into
# one number, so retiring one channel moves it by less than the span this
# fixture reads across the runners; the census gates each fold's coverage
# instead, and it reads the same on every host.
#
# Every builtin the generic walker fold covers, one hot loop per channel.
#
# Without a fold, `hash(x)` / `ord(c)` / `abs(x)` / `min(a, b)` each reach the
# interpreter as `bh_call_fn(builtin, NULL, ...)`, and that residual costs the
# same for all of them: the frame force, the argument rooting, the execution
# context resolution and the gateway signature binding all run before the body
# does.  The fold emits a direct call into the builtin's raw helper, a guard on
# that channel's decline sentinel, and an inline `wrapint` / `wrapfloat` the
# optimizer can keep virtual.
#
# The ratio is the detector here: losing a fold changes no jit-stats counter,
# because the residual it falls back to compiles the same loop.
#
#   hash_int/hash_str  the `Int1` channel — an `i64` result and the
#                      `INT_FOLD_DECLINE` guard.
#   ord_str            the same channel on an operand whose acceptance is a
#                      length, not a type.
#   abs_int/abs_float  one builtin holding two rows, one per result channel.
#   min_max            the `Ref2` channel — the helper returns one of its own
#                      arguments, so nothing is allocated at all.
HASH_N = 16000000
ORD_N = 16000000
ABS_N = 16000000
MINMAX_N = 12000000


def run_hash_int():
    # `hash(int)` is the value itself, so this total is the same everywhere.
    total = 0
    x = 1234567
    for _ in range(HASH_N):
        total += hash(x)
    return total


def run_hash_str():
    # A string's hash is seeded per process, so the digest itself cannot be
    # printed.  Count the iterations that agree with the first one instead:
    # the fold still has to produce the digest, and the count is invariant.
    s = "specialize"
    first = hash(s)
    same = 0
    for _ in range(HASH_N):
        if hash(s) == first:
            same += 1
    return same


def run_ord():
    total = 0
    c = "q"
    for _ in range(ORD_N):
        total += ord(c)
    return total


def run_abs_int():
    total = 0
    x = -7
    for _ in range(ABS_N):
        total += abs(x)
    return total


def run_abs_float():
    total = 0.0
    x = -7.5
    for _ in range(ABS_N):
        total += abs(x)
    return total


def run_min_max():
    total = 0
    a = 3
    b = 9
    for _ in range(MINMAX_N):
        total += min(a, b) + max(a, b)
    return total


print(run_hash_int())
print(run_hash_str())
print(run_ord())
print(run_abs_int())
print(round(run_abs_float(), 6))
print(run_min_max())
