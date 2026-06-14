N = 200000


def make_adder():
    base = 7

    def run(n):
        acc = 0
        i = 0
        while i < n:
            acc = acc + base
            i = i + 1
        return acc

    return run


print(make_adder()(N))
