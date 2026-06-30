def f(types):
    total = 0
    for i in range(2000):
        total = total + types[i]
    return total
types = [1]*300 + [0.5] + [1]*1699
print(f(types))
