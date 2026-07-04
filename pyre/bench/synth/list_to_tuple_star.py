N = 50000


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
