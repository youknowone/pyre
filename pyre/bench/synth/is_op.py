N = 200000


def main():
    a = object()
    b = object()
    acc = 0
    i = 0
    while i < N:
        c = a if (i & 1) == 0 else b
        if c is a:
            acc = acc + 1
        else:
            acc = acc + 2
        if c is not b:
            acc = acc + 10
        i = i + 1
    print(acc)


main()
