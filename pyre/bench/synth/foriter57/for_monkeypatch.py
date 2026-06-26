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

    def stop(self):  # monkeypatched __next__: the iterator now yields nothing
        raise StopIteration

    It.__next__ = stop
    b = run()
    return (a, b)
print(f())
