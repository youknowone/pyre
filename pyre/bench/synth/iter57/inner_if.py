class EvenDown:
    def __init__(self, n):
        self.n = n
    def __iter__(self):
        return self
    def __next__(self):
        self.n = self.n - 1
        if self.n < 0:
            raise StopIteration
        if self.n % 2 == 0:
            return self.n
        return -self.n

total = 0
for _ in range(20000):
    for x in EvenDown(6):
        total += x
print(total)
