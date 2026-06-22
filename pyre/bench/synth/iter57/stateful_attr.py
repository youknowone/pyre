class Down:
    def __init__(self, n):
        self.n = n
    def __iter__(self):
        return self
    def __next__(self):
        if self.n <= 0:
            raise StopIteration
        self.n = self.n - 1   # STORE_ATTR (deferred by #143)
        return self.n

total = 0
for _ in range(20000):
    for x in Down(5):
        total += x
print(total)
