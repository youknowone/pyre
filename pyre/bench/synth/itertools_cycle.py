import itertools

# GET_ITER + FOR_ITER directly over an itertools.cycle iterator. cycle is
# "already an iterator" (its __iter__ returns self) and advances through
# space.next, so the for-loop must classify it on both legs.
out = []
n = 0
for v in itertools.cycle([1, 2, 3]):
    out.append(v)
    n += 1
    if n == 8:
        break
print("forloop:", out)

# cycle over a string, driven by enumerate.
e = []
for i, x in enumerate(itertools.cycle("AB")):
    e.append((i, x))
    if i >= 4:
        break
print("enum:", e)

# explicit next() over a cycle.
c = itertools.cycle([10, 20])
print("next:", [next(c) for _ in range(5)])

# cycle that exhausts its source before replaying the saved items.
total = 0
m = 0
for v in itertools.cycle(range(4)):
    total += v
    m += 1
    if m == 10:
        break
print("range-source sum:", total)
