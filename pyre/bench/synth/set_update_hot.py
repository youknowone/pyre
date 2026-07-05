# N/ITERS are kept small so the wasm backend finishes inside the synthetic
# timeout: wasm runs every guard-exit re-entry through the not-yet-collected
# interpreter allocation path, so the run's wall grows super-linearly in ITERS
# (the pre-existing #62 leak; native dynasm/cranelift stay linear via bridge
# chaining).  The point is to prove the opcode compiles, not to race pypy.
N = 300
ITERS = 500


def run(n, base):
    # `{i, *base}` compiles to BUILD_SET + SET_UPDATE in a while-loop body.
    # Before SET_UPDATE was lowered, its abort_permanent marker declined
    # the whole loop.
    total = 0
    i = 0
    while i < n:
        s = {i, *base}
        total += len(s)
        i += 1
    return total


def main():
    base = [10, 20, 30]
    total = 0
    for _ in range(ITERS):
        total += run(N, base)
    print(total)


main()
