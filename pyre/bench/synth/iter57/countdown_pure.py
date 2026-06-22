class CountDown:
    def __init__(self, n):
        self.n = n
    def __iter__(self):
        return self
    def __next__(self):
        if self.n <= 0:
            raise StopIteration
        v = self.n
        self.n = self.n - 1
        return v

total = 0
for _ in range(20000):
    for x in CountDown(5):
        total += x
print(total)
