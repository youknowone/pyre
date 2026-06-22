class Boom:
    def __init__(self, n):
        self.n = n
    def __iter__(self):
        return self
    def __next__(self):
        self.n = self.n - 1
        if self.n == 0:
            raise ValueError("boom")
        if self.n < -3:
            raise StopIteration
        return self.n

seen = []
for _ in range(20000):
    try:
        for x in Boom(5):
            seen.append(x)
    except ValueError as e:
        seen.append(str(e))
print(len(seen), seen[:8])
