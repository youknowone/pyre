# Hot read of an unboxed FLOAT attribute. mapdict keeps a float-valued
# attribute in raw storage, so `self.x` in a loop leaves exactly one residual:
# `jit_mapdict_unboxed_read_f`, whose recorded descr is `[Ref, Int, Int] ->
# Float`. On wasm that is a MIXED signature -- `(i64, i64, i64) -> f64` -- and
# the backend's arity-keyed `call_indirect` family only covers uniform word
# shapes, so before the callee was vouched every iteration crossed out to the
# host `jit_call` trampoline: 398958 crossings for 400000 iterations, 100% of
# them this one callee.
#
# The int-valued twin below is the control that makes the attribution
# unambiguous: it is the same loop over the same class shape, its residual is
# the `(i64, i64, i64) -> i64` sibling, and it crosses ZERO times because a
# uniform word signature already lowers in-module. Native dynasm shows no
# float-vs-int gap at all, so anything left here is the crossing and not the
# read.
#
# No `max-wasm-ratio` header: that directive is an ALLOWANCE carved out of
# `WASM_MAX_DYNASM_RATIO`, and this fixture wants the default rather than an
# exemption from it. The default is what holds the change in place -- the read
# measures 1.2x dynasm now, and losing the vouch puts a host trampoline back on
# every iteration at roughly 168 ns each, which is 34x the int-attribute twin
# and far past 4.0. The fixture is sized so dynasm's execution-only time clears
# FLOOR_GATE_MIN_BASELINE_S; below that the gate declines to judge and the
# fixture would gate nothing.
#
# The remaining legs pin the shapes the unboxed read must NOT be asked to
# answer, each written so a wrong answer is a wrong value rather than a silent
# pass:
#
#   * storing a non-float into the same attribute REVOKES unboxing for that
#     slot, and the loop after it must read the boxed value;
#   * a second instance of the same class shares the storage layout, so the
#     helper's `storageindex`/`listindex` constants must still address the
#     right slot on it;
#   * a subclass adding its own attribute gets a different layout, and reading
#     the inherited one through it must not reuse the parent's indices.


class Point:
    def __init__(self, x):
        self.x = x


class Tagged(Point):
    def __init__(self, x, tag):
        Point.__init__(self, x)
        self.tag = tag


def read_float_hot(n, p):
    acc = 0.0
    for _ in range(n):
        acc += p.x
    return acc


def read_int_hot(n, p):
    acc = 0
    for _ in range(n):
        acc += p.x
    return acc


def revoked_shapes(n):
    acc = 0.0
    for _ in range(n):
        p = Point(1.5)
        acc += p.x
        # Unboxing for this slot is revoked by the non-float store; the read
        # after it goes through the boxed strategy.
        p.x = "s"
        acc += len(p.x)
        other = Point(0.25)
        acc += other.x
        sub = Tagged(0.5, 7)
        acc += sub.x + sub.tag
    return acc


print(read_float_hot(4000000, Point(1.5)))
print(read_int_hot(4000000, Point(2)))
print(revoked_shapes(20000))
