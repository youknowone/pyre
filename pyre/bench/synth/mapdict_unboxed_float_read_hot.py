# Hot read of an unboxed FLOAT attribute. mapdict keeps a float-valued
# attribute in raw storage, and `self.x` in a loop now reads it as the three
# loads it is -- the instance's storage block, the slot holding the
# attribute's raw list, and the item -- so the loop leaves NO residual. Its
# optimized body is 14 ops against the 25 (six of them calls) it carried while
# the read went through `jit_mapdict_unboxed_read_f`.
#
# That helper is why this file exists, and the reason is now historical. Its
# recorded descr was `[Ref, Int, Int] -> Float`, which on wasm is a MIXED
# signature -- `(i64, i64, i64) -> f64`. The backend's arity-keyed
# `call_indirect` family covers only uniform word shapes, so before the callee
# was vouched in `eval.rs`'s faithful-residual list every iteration crossed out
# to the host `jit_call` trampoline: 398958 crossings for 400000 iterations,
# 100% of them this one callee. The int-valued twin below was the control that
# made the attribution unambiguous -- same loop, same class shape, a
# `(i64, i64, i64) -> i64` sibling that a uniform word signature already lowers
# in-module, and ZERO crossings.
#
# Neither loop reaches a residual any more, so NEITHER gates that vouch. The
# entries stay named in `eval.rs` for as long as the helpers do; what this file
# still measures is the read and store themselves against pypy.
#
# No `max-wasm-ratio` header: that directive is an ALLOWANCE carved out of
# `WASM_MAX_DYNASM_RATIO`, and this fixture wants the default rather than an
# exemption from it. The fixture is sized so dynasm's execution-only time
# clears FLOOR_GATE_MIN_BASELINE_S; below that the gate declines to judge and
# the fixture would gate nothing.
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
