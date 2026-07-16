# Nested-loop tracer correctness gate.  Every case prints multiple checksums
# so a dropped or duplicated inner iteration cannot be masked by a count.
MOD = 1000003


def rectangular(outer, inner):
    s = 0; sq = 0; cu = 0
    n = 0
    while n < outer:
        for i in range(inner):
            v = (i * 7 + n * 3 + 1) % MOD
            s = (s + v) % MOD
            sq = (sq + v * v) % MOD
            cu = (cu + v * v * v) % MOD
        n += 1
    return s, sq, cu


def triangular(m):
    # The inner trip count depends on the outer value: this catches an outer
    # exit-prediction guard being resumed in an inner-loop trace.
    s = 0; sq = 0
    for i in range(m):
        j = 0
        while j < i:
            v = (i * j + 5) % MOD
            s = (s + v) % MOD
            sq = (sq + v * v) % MOD
            j += 1
    return s, sq


def for_in_for(outer, inner):
    s = 0; sq = 0
    for a in range(outer):
        for b in range(inner):
            v = (a * b + a + b) % MOD
            s = (s + v) % MOD
            sq = (sq + v * v) % MOD
    return s, sq


def triple_nest(a_n, b_n, c_n):
    s = 0
    for a in range(a_n):
        for b in range(b_n):
            for c in range(c_n):
                s = (s + a * b * c + a + b + c) % MOD
    return s


print("rect_40_50", rectangular(40, 50))
print("rect_50_40", rectangular(50, 40))
print("rect_1000_200", rectangular(1000, 200))
print("tri_200", triangular(200))
print("tri_523", triangular(523))
print("fif_300_211", for_in_for(300, 211))
print("triple_40_30_20", triple_nest(40, 30, 20))
