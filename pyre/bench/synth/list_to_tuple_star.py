# pyre-check: max-pypy-ratio=36
# N is sized so pypy clears `FLOOR_GATE_MIN_BASELINE_S`.  At 100000
# iterations the run finishes inside startup, pypy exec is clamped, and a
# ratio is not a measurement.  3200000 iterations land pypy near 0.08s.
# Local dynasm reads 17x; 36 leaves room for cranelift and a slower host.
# `loops_compiled` stays 1, and the jitstats baseline still gates that shape.
N = 3200000


def main():
    # `(*a, i, *b)` compiles to LIST_TO_TUPLE (CALL_INTRINSIC_1) after the
    # star-unpack BUILD_LIST/LIST_EXTEND.  Lowered but latent — the enclosing
    # star-unpack construct pulls in other unported ops, so no demonstrable
    # speedup; this bench guards LIST_TO_TUPLE output correctness.
    a = [1, 2]
    b = [3, 4, 5]
    acc = 0
    for i in range(N):
        t = (*a, i, *b)
        acc += len(t) + t[0] + t[-1]
    print(acc)


main()
