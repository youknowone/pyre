class It:
    def __init__(self, n):
        self.i = 0
        self.n = n
    def __iter__(self):
        return self
    def __next__(self):
        if self.i >= self.n:
            raise StopIteration
        v = self.i
        self.i += 1
        return v
def run():
    s = 0
    for x in It(50):
        s += x
    return s
def f():
    a = run()
    It.__next__ = lambda self: (_ for _ in ()).throw(StopIteration)  # now yields nothing
    b = run()
    return (a, b)
print(f())
