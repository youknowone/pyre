class HotBoom:
    def __init__(self, n):
        self.i = 0
        self.n = n
    def __iter__(self):
        return self
    def __next__(self):
        self.i += 1
        if self.i > self.n:
            raise ValueError("boom")
        return self.i
def f():
    total = 0
    try:
        for x in HotBoom(500):
            total += x
    except ValueError:
        total += 1000000
    return total
print(f())
