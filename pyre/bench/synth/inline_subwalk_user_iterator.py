# gh#495 guard: inlined callee consumes user iterator whose next mutates state.
# FOR loop over user iterator INSIDE branch-bearing inlined callee; __next__ mutates shared counter
N = 30000
class It:
    def __init__(self): self.pos = 0; self.lim = 3
    def __iter__(self):
        self.n = 0
        return self
    def __next__(self):
        self.n = self.n + 1
        if self.n > self.lim:
            raise StopIteration
        self.pos = self.pos + 1
        return self.n
def step(it, d, k):
    if k < 0:
        return 0
    if k in d:
        s = 0
        for x in it:          # FOR_ITER over user iterator inside inlined callee branch
            s += x
        return s
    return -1
def run():
    d = {1:100,2:200,3:300}; acc=0; it=It(); i=0
    while i < N:
        k = i % 5
        v = step(it, d, k)
        acc += v
        i += 1
    return acc, it.pos
print(run())
