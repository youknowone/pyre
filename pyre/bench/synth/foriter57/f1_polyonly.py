def f(data):
    total = 0
    for x in data:
        total = total + x
    return total
data = list(range(50000)) + [0.5] + list(range(50000))
print(f(data))
