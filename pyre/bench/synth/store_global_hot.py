# STORE_GLOBAL inside a hot loop: a `global`-declared function reassigns
# module globals every iteration.  The compiled per-CodeObject jitcode
# walks the `store_global` residual instead of an abort_permanent marker,
# so the hot body JIT-compiles rather than declining to the trait leg.
# A conditional exercises a branch between the two STORE_GLOBAL sites.
# Output is verified against CPython/PyPy.
N = 200000

counter = 0
total = 0


def run(n):
    global counter, total
    i = 0
    while i < n:
        counter = counter + 1
        if i % 2 == 0:
            total = total + i
        else:
            total = total - 1
        i = i + 1
    return total


def main():
    r = run(N)
    print("counter", counter)
    print("total", total)
    print("r", r)


main()
