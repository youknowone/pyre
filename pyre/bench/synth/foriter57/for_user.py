class Counter:
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
def f():
    s = 0
    for x in Counter(500):
        s += x
    return s
print(f())
