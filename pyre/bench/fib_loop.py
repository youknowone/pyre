MOD = 1000000007


def fib(n):
    a = 0
    b = 1
    i = 0
    while i < n:
        t = a + b
        a = b
        b = t
        i = i + 1
    return b


def main():
    for n in [20000, 40000, 60000, 80000, 100000]:
        print(fib(n) % MOD)

main()
