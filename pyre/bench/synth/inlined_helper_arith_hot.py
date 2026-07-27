# pyre-check: max-pypy-ratio=12
# One-line arithmetic helpers called from a hot `for` loop body. The inline
# lever gates a FOR_ITER-in-flight callee on `fbw_callee_body_replay_safety`:
# a body whose only residual is a BINARY_OP the walker will specialize to a
# native op leaves nothing to replay and is admitted, anything else is Dirty
# and the whole callee stays a per-iteration residual call.
#
# The accepted tag set is every tag a specialization table lowers with no
# runtime decline left. Add / Subtract / Multiply are in both the int
# (IntAddOvf / IntSubOvf / IntMulOvf) and float (FloatAdd / FloatSub /
# FloatMul) tables; And / Or / Xor are in the int table only. Both tables key
# each in-place tag to the same arm as its plain form, so `a += i` is admitted
# exactly like `a + i`. Each loop below pins one tag, so a tag dropping out of
# the set shows up here as the callee reappearing as a residual and the ratio
# blowing past the gate.
#
# The divide / remainder / shift / power tags are deliberately NOT in the set
# — each can still decline (zero divisor, out-of-range shift, nan/inf base) and
# leave the residual behind — and they are kept out of this file so the gate
# measures the admitted paths rather than cases that are meant to stay slow.
# Callees are named directly rather than passed in, so the call site keeps a
# constant callee. Output verified against CPython/PyPy.
#
# N is sized so PyPy's startup-subtracted user-CPU stays well clear of its own
# empty-program startup. The gate divides by that difference, so a workload
# finishing near startup makes the denominator a residue of two similar
# measurements: at N=200000 it lands on the 5ms floor and unchanged code
# reports anywhere from 6x to 22x depending on runner speed. At this N the
# ratio is 2.5x (dynasm) / 3.1x (cranelift), which is the value the gate is
# meant to bound.
N = 4000000


def add_body(a, i):
    return a + i


def sub_body(a, i):
    return a - i


def mul_body(a, i):
    return a + i * 2


def iadd_body(a, i):
    a += i
    return a


def isub_body(a, i):
    a -= i
    return a


def imul_body(a, i):
    b = i
    b *= 2
    return a + b


def and_body(a, i):
    return a + (i & 255)


def or_body(a, i):
    return a + (i | 1)


def xor_body(a, i):
    return a + (i ^ 5)


def ibit_body(a, i):
    b = i
    b &= 255
    b |= 1
    b ^= 5
    return a + b


def mixed_body(a, i):
    return a + i * 3 - 1


def float_body(a, i):
    return a + i * 0.5


def float_iadd_body(a, i):
    a += i * 0.25
    return a


def run_add(n):
    s = 0
    for i in range(n):
        s = add_body(s, i)
    return s


def run_sub(n):
    s = 0
    for i in range(n):
        s = sub_body(s, i)
    return s


def run_mul(n):
    s = 0
    for i in range(n):
        s = mul_body(s, i)
    return s


def run_iadd(n):
    s = 0
    for i in range(n):
        s = iadd_body(s, i)
    return s


def run_isub(n):
    s = 0
    for i in range(n):
        s = isub_body(s, i)
    return s


def run_imul(n):
    s = 0
    for i in range(n):
        s = imul_body(s, i)
    return s


def run_and(n):
    s = 0
    for i in range(n):
        s = and_body(s, i)
    return s


def run_or(n):
    s = 0
    for i in range(n):
        s = or_body(s, i)
    return s


def run_xor(n):
    s = 0
    for i in range(n):
        s = xor_body(s, i)
    return s


def run_ibit(n):
    s = 0
    for i in range(n):
        s = ibit_body(s, i)
    return s


def run_mixed(n):
    s = 0
    for i in range(n):
        s = mixed_body(s, i)
    return s


def run_float(n):
    s = 0.0
    for i in range(n):
        s = float_body(s, i)
    return s


def run_float_iadd(n):
    s = 0.0
    for i in range(n):
        s = float_iadd_body(s, i)
    return s


print(run_add(N))
print(run_sub(N))
print(run_mul(N))
print(run_iadd(N))
print(run_isub(N))
print(run_imul(N))
print(run_and(N))
print(run_or(N))
print(run_xor(N))
print(run_ibit(N))
print(run_mixed(N))
print(run_float(N))
print(run_float_iadd(N))
