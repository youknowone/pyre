# Nested FOR_ITER with range iterators only (no list).
# The outer for-loop drives an inner for-loop — both are range().
# Tests that nested FOR_ITER with homogeneous builtin iterators JITs
# correctly and without perf regression.

def main():
    total = 0
    n = 0
    while n < 1000:
        for i in range(200):
            total += i * n
        n += 1
    return total

result = main()
print(result)
# Expected: sum(i*n for n in range(1000) for i in range(200))
#         = sum(n for n in range(1000)) * sum(i for i in range(200))
#         = 499500 * 19900 = 9940050000
