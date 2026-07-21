# complex() falls back to __index__ when a value defines neither __complex__
# nor __float__, matching float()'s coercion. Output verified against
# CPython/PyPy.
N = 40000


class Idx:
    def __index__(self):
        return 3


def main():
    obj = Idx()
    acc = 0.0
    for _ in range(N):
        c = complex(obj)
        acc += c.real
    print(complex(obj), acc == N * 3.0)


main()
