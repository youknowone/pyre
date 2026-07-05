# FOR_ITER body with LOAD_GLOBAL: the JIT must handle module-global
# reads inside for-loop bodies correctly.

SCALE = 3

def main():
    total = 0
    n = 0
    while n < 20000:
        for x in range(10):
            total += x * SCALE
        n += 1
    return total

result = main()
print(result)
# Expected: 20000 * sum(x*3 for x in range(10)) = 20000 * 135 = 2700000
