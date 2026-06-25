def gen(n):
    i = 0
    while i < n:
        yield i
        i += 1
def f():
    s = 0
    for x in gen(500):
        s += x
    return s
print(f())
