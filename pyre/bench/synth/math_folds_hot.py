# pyre-check: max-pypy-ratio=5
# pyre-check: skip-cpython
# pyre-check: max-wasm-ratio=13
# Fitted to the highest reading observed plus 15%: darwin-arm64 reads
# 8.1-9.0x across five runs at load 11 and 11.3x during a load spike.  The wasm
# side is slower for a structural reason rather than a regression: `pymath`
# reaches the platform libm on native and its pure-Rust `libm` fallback in the
# guest, and the guest also pays an errno-classifying wrapper the native build
# folds away.  Measured on the same fold machinery and the same loop, wasm
# runs 2M folded `log` (which lowers to `x.ln()`) in 0.09s and 2M folded `exp`
# (which goes through `pymath`) in 0.24s.  Both backends fold; only the
# operation underneath differs.
# The ratio is the detector here: losing a fold changes no jit-stats counter,
# because the residual it falls back to compiles the same loop.  Measured on
# an idle darwin-arm64 box against pypy 0.33s — every fold 0.63s, the generic
# float/isclose folds suppressed 2.71s, all folds suppressed 33.4s.  The gate
# sits between the first two.
# Every `math` primitive the walker folds, one hot loop per fold shape.  A
# residual `bh_call_fn(builtin, NULL, x)` costs an argument tuple, a
# `W_FloatObject` allocation and a full builtin dispatch per iteration; each
# specialization instead unboxes the operands, emits the raw operation, and
# leaves the result box virtualizable.
#
#   sqrt         `try_walker_specialize_math_sqrt` — `x >= 0` and `isfinite(x)`
#                pin the two `ll_math_sqrt` branches, then a pure
#                `CALL_F(sqrt_nonneg_jit)` with no result guard.
#   log/cos/sin  `try_walker_specialize_math_log_trig` — same shape, one
#                domain guard each.
#   fabs         `try_walker_specialize_math_fabs` — a single `FloatAbs`;
#                `fabs` raises for no input, so it carries no domain guard.
#   floor/ceil/  `try_walker_specialize_math_round_to_int` — guard the operand
#   trunc        into the signed machine range, round, `CastFloatToInt`.
#   isclose      `try_walker_specialize_math_isclose` — one pure `CALL_I`
#                into a total helper, whose truth the branch's own guard
#                already pins, so it carries no result guard.
#   the rest     `try_walker_specialize_math_float{1,2}` — one pure elidable
#                `CALL_F` into the function's raw helper plus a finite-result
#                guard.  The helper reports every raising direction as NaN, so
#                the guard alone carries the domain: `exp` overflowing,
#                `atanh` outside (-1, 1) and `pow(0.0, -2.0)` all resume in
#                the builtin and raise there.
#
# A rebound callable, a numeric subclass, or an operand outside the folded
# domain keeps the residual in every case.
import math

# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and declines the baseline as too small.
SQRT_N = 8000000
LOG_TRIG_N = 1600000
FABS_N = 10000000
ROUND_N = 10000000
UNARY_N = 1500000
BINARY_N = 2000000
ISCLOSE_N = 4000000


def run_sqrt():
    total = 0.0
    for i in range(SQRT_N):
        total += math.sqrt(float(i))
    return total


def run_log_trig():
    total = 0.0
    for i in range(LOG_TRIG_N):
        x = 1.0 + float(i % 97) / 97.0
        total += math.log(x) + math.cos(x) + math.sin(x)
    return total


def run_fabs():
    total = 0.0
    for i in range(FABS_N):
        total += math.fabs(float(i) - 6000000.0)
    return total


def run_round_to_int():
    total = 0
    for i in range(ROUND_N):
        x = float(i) * 0.5 - 1000.0
        total += math.floor(x) + math.ceil(x) + math.trunc(x)
    return total


def run_unary():
    total = 0.0
    for i in range(UNARY_N):
        x = float(i % 71) / 128.0
        total += math.exp(x) + math.tan(x) + math.atan(x)
        total += math.tanh(x) + math.log1p(x) + math.degrees(x)
    return total


def run_binary():
    total = 0.0
    for i in range(BINARY_N):
        x = 1.0 + float(i % 53) / 53.0
        y = 1.0 + float(i % 31) / 31.0
        total += math.pow(x, y) + math.fmod(x, y)
        total += math.copysign(x, y) + math.atan2(x, y)
    return total


def run_isclose():
    # The result has to decide a branch and nothing else: a bool that escapes
    # keeps the residual, because pinning it would bail on every re-entry with
    # the other truth.
    # The perturbation stays well inside the default `rel_tol=1e-09`, so the
    # branch resolves the same way every iteration and the trace measures the
    # fold rather than a bridge.
    hits = 0
    for i in range(ISCLOSE_N):
        x = 1.0 + float(i % 97) / 1e12
        if math.isclose(x, 1.0):
            hits += 1
    return hits


print(round(run_sqrt(), 6))
print(round(run_log_trig(), 6))
print(round(run_fabs(), 6))
print(run_round_to_int())
print(round(run_unary(), 6))
print(round(run_binary(), 6))
print(run_isclose())
