# gh#495 guard: constant-int-returning callee mutation must not be replayed or dropped.
N = 30000
class C:
    def __init__(self): self.pos = 0
def step(c):
    c.pos = c.pos + 1
    return 7
def run():
    c = C(); acc = 0; i = 0
    while i < N:
        acc = acc + step(c); i = i + 1
    return acc, c.pos
print(run())
