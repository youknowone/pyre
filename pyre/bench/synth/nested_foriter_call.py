# Nested FOR_ITER with a function call in the inner body.
# Tests CALL inside nested FOR_ITER body — the Layer 2 defense
# (inline sub-walk decline) must handle nested context.

def add(a, b):
    return a + b

def main():
    total = 0
    n = 0
    while n < 1000:
        for j in range(200):
            total = add(total, n * j)
        n += 1
    return total

result = main()
print(result)
# Expected: sum(n*j for n in range(1000) for j in range(200))
#         = sum(n for n in range(1000)) * sum(j for j in range(200))
#         = 499500 * 19900 = 9940050000
