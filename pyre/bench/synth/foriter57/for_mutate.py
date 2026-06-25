def f():
    seen = []
    for x in range(500):
        seen.append(x)
    return len(seen)
print(f())
