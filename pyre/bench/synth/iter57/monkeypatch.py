class It:
    def __init__(self):
        self.n = 0
    def __iter__(self):
        return self
    def __next__(self):
        self.n = self.n + 1
        if self.n > 100000:
            raise StopIteration
        return self.n

def patched(self):
    self.n = self.n + 1
    if self.n > 100000:
        raise StopIteration
    return self.n * 2

it = It()
total = 0
count = 0
for x in it:
    total += x
    count += 1
    if count == 50000:
        It.__next__ = patched   # invalidate the method cache mid-loop
print(total, count)
