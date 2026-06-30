def f(data):
    total = 0
    for x in data:
        total = total + x
    return total
data = [1]*400 + [0.5] + [1]*100
print(f(data))
