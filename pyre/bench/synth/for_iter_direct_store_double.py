# pyre-check: max-pypy-ratio=0.5
try:
    import pypyjit
except ImportError:
    pypyjit = None


def set_trace_limit(n):
    if pypyjit is not None:
        pypyjit.set_param("trace_limit=%d,threshold=1,function_threshold=1" % n)


set_trace_limit(40)

errs = list(map(lambda i: OSError(1, str(i)), range(20000)))


def mutate_all(items):
    total = 0
    for e in items:
        e.errno = e.errno + 1
        tmp = total
        del tmp
        a = total + 1
        b = a + 2
        c = b + 3
        d = c + 4
        f = d + 5
        g = f + 6
        h = g + 7
        i = h + 8
        j = i + 9
        k = j + 10
        m = k + 11
        n = m + 12
        o = n + 13
        p = o + 14
        q = p + 15
        r = q + 16
        s = r + 17
        t = s + 18
        total = t & 1023
    return total


total = mutate_all(errs)
print(sum(map(lambda e: e.errno, errs)), total)
