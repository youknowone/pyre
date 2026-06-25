def f():
    s = 0
    for x in range(100000):
        s += x
    return s
print(f())
