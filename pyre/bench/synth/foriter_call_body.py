# FOR_ITER body with a CALL: the JIT must handle calls inside for-loop
# bodies correctly without replaying the last iteration on deopt.

def g(x):
    return x * 2

def main():
    total = 0
    n = 0
    while n < 20000:
        for x in range(10):
            total += g(x)
        n += 1
    return total

result = main()
print(result)
# Expected: 20000 * sum(2*x for x in range(10)) = 20000 * 90 = 1800000
