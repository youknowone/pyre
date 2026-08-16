# pyre-check: max-pypy-ratio=6
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
# loop, and the recursion is a callee it reaches; that ordering is what used to
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
    while i < 300000:
        total = (total + step(8, i)) % MOD
        i += 1
    print("recursion_from_loop", total)


main()
