# N/ITERS are kept small so the wasm backend finishes inside the synthetic
# timeout: wasm runs every guard-exit re-entry through the not-yet-collected
# interpreter allocation path, so the run's wall grows super-linearly in ITERS
# (the pre-existing #62 leak; native dynasm/cranelift stay linear via bridge
# chaining).  The point is to prove the opcode compiles, not to race pypy.
N = 300
ITERS = 500


def run(n, extra):
    # `{"k": i, **extra}` compiles to BUILD_MAP + DICT_UPDATE in a
    # while-loop body.  Before DICT_UPDATE was lowered, its abort_permanent
    # marker declined the whole loop.
    total = 0
    i = 0
    while i < n:
        d = {"k": i, **extra}
        total += d["k"] + d["x"]
        i += 1
    return total


def main():
    extra = {"x": 7}
    total = 0
    for _ in range(ITERS):
        total += run(N, extra)
    print(total)


main()
