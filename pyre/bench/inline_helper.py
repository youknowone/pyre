# 150e6 so leftover pyre startup is a small fraction of the 1.5x pypy budget.
def add(a, b):
    return a + b

def mul(a, b):
    return a * b

def square(x):
    return mul(x, x)

def compute(x):
    return add(square(x), x)

def main():
    s = 0
    i = 0
    while i < 150000000:
        s = add(s, compute(i)) % 1000000007
        i = add(i, 1)
    print(s)

main()
