# Nested FOR_ITER with a function call in the inner body.
# Tests CALL inside nested FOR_ITER body — the Layer 2 defense
# (inline sub-walk decline) must handle nested context.

def add(a, b):
    return a + b

def main():
    total = 0
    n = 0
    while n < 100:
        for j in range(200):
            total = add(total, n * j)
        n += 1
    return total

result = main()
print(result)
# Expected: sum(n*j for n in range(100) for j in range(200))
#         = sum(n for n in range(100)) * sum(j for j in range(200))
#         = 4950 * 19900 = 98505000
