# pyre-check: max-pypy-ratio=6.3
# FOR_ITER body with LOAD_GLOBAL: the JIT must handle module-global
# reads inside for-loop bodies correctly.

SCALE = 3

def main():
    total = 0
    n = 0
    while n < 4000000:
        for x in range(10):
            total += x * SCALE
        n += 1
    return total

result = main()
print(result)
# Expected: 4000000 * sum(x*3 for x in range(10)) = 4000000 * 135 = 540000000
