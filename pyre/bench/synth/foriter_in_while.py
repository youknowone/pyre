# FOR_ITER inside while-loop: the common real-world pattern.
# The while-loop is the JIT entry; the for-loop body must handle
# the FOR_ITER liveness scoping correctly.

def main():
    total = 0
    n = 0
    while n < 10000:
        for x in range(100):
            total += x
        n += 1
    return total

result = main()
print(result)
# Expected: 10000 * sum(range(100)) = 10000 * 4950 = 49500000
