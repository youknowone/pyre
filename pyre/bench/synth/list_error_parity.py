# list error-message parity where 3.14 and PyPy agree (check.py oracle is
# PyPy).  list.index uses "%R is not in list" (the value's repr); list.pop
# distinguishes empty vs out-of-range.  list.remove is intentionally omitted:
# 3.14 says "list.remove(x): x not in list" but PyPy says "... (): N is ...",
# a divergence that cannot be asserted against the PyPy oracle.
def cap(fn):
    try:
        return ("ok", fn())
    except (ValueError, IndexError, TypeError) as e:
        return (type(e).__name__, str(e))


def drive():
    out = []
    out.append(("index_missing", cap(lambda: [1, 2, 3].index(9))))
    out.append(("index_str", cap(lambda: [1, 2].index("z"))))
    out.append(("index_missing_range", cap(lambda: [1, 2, 1].index(1, 2))))
    out.append(("pop_empty", cap(lambda: [].pop())))
    out.append(("pop_oob", cap(lambda: [1, 2].pop(9))))
    # hot index-miss so a compiled trace exercises the raise path
    hits = 0
    n = 0
    while n < 20000:
        try:
            [1, 2, 3].index(9)
        except ValueError:
            hits += 1
        n += 1
    out.append(("hot_index_miss", hits))
    return out


for row in drive():
    print(row)
