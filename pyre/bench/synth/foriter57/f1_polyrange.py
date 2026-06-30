def f(types):
    total = 0
    for i in range(100001):
        total = total + types[i]
    return total
types = [1]*50000 + [0.5] + [1]*50000
print(f(types))
