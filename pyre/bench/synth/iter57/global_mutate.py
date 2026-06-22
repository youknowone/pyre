hits = 0
sink = []

class Tick:
    def __init__(self, n):
        self.n = n
    def __iter__(self):
        return self
    def __next__(self):
        global hits
        if self.n <= 0:
            raise StopIteration
        self.n = self.n - 1
        hits = hits + 1        # STORE_GLOBAL (NOT deferred)
        sink.append(self.n)    # LIST_APPEND (residual shared-heap)
        return self.n

for _ in range(20000):
    for _x in Tick(5):
        pass
print(hits, len(sink), sum(sink))
