def f(data):
    total = 0
    for x in data:
        total = total + x
    return total
data = tuple([1]*3000 + [0.5] + [1]*3000)
print(f(data))
