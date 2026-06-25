def f():
    total = 0
    for i, x in enumerate([10, 20, 30, 40]):
        total += i * x
    return total
print(f())
