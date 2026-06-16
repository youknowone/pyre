# Builds a container (tuple / list / set / dict / str) AFTER a hot loop and
# returns it.  The loop-exit guard failure resumes via the blackhole, which
# then walks the forward path through the container-build residual
# (new_array_clear + setarrayitem_gc_r + new*_from_array).  Unlike
# build_tuple_large.py — which builds inside the traced loop body — the build
# here is reached only on the resume path, exercising the blackhole builder's
# coverage of the GC array-build family.
N = 2000000


def build_tuple(n):
    s = 0
    i = 0
    while i < n:
        s = s + i
        i = i + 1
    return s, i, n - i


def build_list(n):
    s = 0
    i = 0
    while i < n:
        s = s + (i & 7)
        i = i + 1
    return [s, i, s + i]


def build_set(n):
    s = 0
    i = 0
    while i < n:
        s = s + (i % 5)
        i = i + 1
    return {s, i, s - i}


def build_map(n):
    s = 0
    i = 0
    while i < n:
        s = s + (i & 3)
        i = i + 1
    return {"s": s, "i": i}


def build_str(n):
    s = 0
    i = 0
    while i < n:
        s = s + 1
        i = i + 1
    return f"<{s}:{i}>"


def main():
    a, b, c = build_tuple(N)
    lst = build_list(N)
    st = build_set(N)
    mp = build_map(N)
    text = build_str(N)
    print(a, b, c)
    print(lst)
    print(sorted(st))
    print(mp["s"], mp["i"])
    print(text)


main()
