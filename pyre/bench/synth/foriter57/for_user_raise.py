class Boom:
    def __init__(self):
        self.i = 0
    def __iter__(self):
        return self
    def __next__(self):
        self.i += 1
        if self.i == 6:
            raise ValueError("boom")
        return self.i
def f():
    total = 0
    try:
        for x in Boom():
            total += x
    except ValueError:
        total += 1000
    return total
print(f())
