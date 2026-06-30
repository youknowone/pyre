def f(types):
    total = 0
    for i in range(2000):
        total = total + types[i]
    return total
types = [1]*301 + [0.5]*1699   # flips at 301 then stays float
print(f(types))
