# Nested FOR_ITER: outer while-loop drives a nested for-loop pair.
# The inner loop's range() iterator is a builtin (no user frame).
# Verifies the JIT handles nested FOR_ITER correctly.

def main():
    total = 0
    n = 0
    while n < 20000:
        for x in [1, 2, 3, 4, 5]:
            for y in range(x):
                total += y
        n += 1
    return total

result = main()
print(result)
# Expected: 20000 * sum(sum(range(x)) for x in [1,2,3,4,5]) = 20000 * 20 = 400000
