# pyre-check: max-pypy-ratio=4.2
# pyre-check: max-wasm-ratio=6.1
#
# Ceiling 6 derived floor 1.0x; after N=4300000, macos dynasm reads 0.8x
# (PR 1736: exec 0.21s vs pypy 0.26s). 4.2 keeps ubuntu dynasm 1.1x and
# cranelift 1.4x under the ceiling and drops the floor to 0.7x.
#
# The walker stops unrolling `step` two frames past `FBW_MAX_INLINE_RECURSION`,
# so each loop iteration is a residual recursive call. wasm compiles one
# module per residual trace; ubuntu-24.04 measured 5.0x on main and 5.3x
# here, so 6.1x is the higher reading plus WASM_RATIO_FIT_HEADROOM (15%).
#
# A recursion deeper than the inline unroll bound, driven from a loop body.
#
# `step` recurses nine frames deep, two past `FBW_MAX_INLINE_RECURSION`, so the
# walker stops unrolling it and the call has to leave the inline route. What
# makes this shape distinct from every other recursion fixture is where the
# call sits: the driver is a `while` body that keeps `total` — a loop-carried
# operand — on the value stack underneath it.
#
# `fib_recursive` and `selfrec_bridge_nontail_promote` do not cover this. There
# the recursion is itself the hot thing, so the callee owns a compiled loop
# before any non-inline decision is taken. Here the hot thing is the caller's
# loop, and the recursion is a callee it reaches; that ordering is what can
# leave the call as an interpreter residual for the rest of the run, one frame
# build and one entry bridge per recursive call. `recursive_call_frame_relocation`
# holds the neighbouring case, a recursion under a `FOR_ITER` iterator, which
# stays on the residual path deliberately.
#
# `step` carries the accumulator down rather than returning into an addition, so
# the recursion is a tail call and the caller's stack under it holds only
# `total`. Arguments stay exact machine integers and the modulus keeps the
# result in range, so nothing here promotes to a long.
MOD = 1000003


def step(n, acc):
    if n <= 0:
        return acc
    return step(n - 1, acc + n)


def main():
    total = 0
    i = 0
    # Sized so pypy's own execution clears Windows `FLOOR_GATE_MIN_BASELINE_S`
    # (~0.16s).  At 300000 the baseline sat in the `?` band and the same
    # binary read 2x-12x.
    while i < 4300000:
        total = (total + step(8, i)) % MOD
        i += 1
    print("recursion_from_loop", total)


main()
