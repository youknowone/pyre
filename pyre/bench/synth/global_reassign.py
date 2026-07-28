# pyre-check: max-pypy-ratio=14
# Exercises JIT global-cache invalidation, for both cell kinds.
#
# `run_int` reads a module global reassigned to another int between calls: the
# compiled loop must observe each new value (quasi-immutable version
# invalidation), not a stale const-folded value.
#
# `run_float` reads a global reassigned to a NON-int object.  A float reassign
# goes through write_cell -> ObjectMutableCell, so the compiled loop folds the
# cell pointer (quasi-immutable version) and must read cell.w_value LIVE via
# GetfieldGcR — an in-place reassign of the same cell does not bump the
# version, so a stale const-fold would return the previous value.
#
# Correct output is verified against CPython/PyPy.
N = 300000


def run_int():
    s = 0
    for _ in range(N):
        s += G
    return s


def run_float():
    s = 0.0
    for _ in range(N):
        s += GF
    return s


G = 3
a = run_int()
G = 7
b = run_int()
G = 11
c = run_int()
G = 2
d = run_int()
print(a, b, c, d)

GF = 3.0
e = run_float()
GF = 7.0
f = run_float()
GF = 11.0
g = run_float()
GF = 2.0
h = run_float()
print(e, f, g, h)
